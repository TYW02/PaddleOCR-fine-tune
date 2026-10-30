#!/usr/bin/env python3
"""Train the project's small StrokeRestorer architecture on TextZoom train1 + train2.

This is a controlled data experiment, not an implementation of TSRN.  The official
TextZoom test/easy, test/medium and test/hard LMDBs are never opened here.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from io import BytesIO
from pathlib import Path

import lmdb
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_accepted_pairs import StrokeRestorer, gradient_loss  # noqa: E402


REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO / "data" / "textzoom"
DEFAULT_OUTPUT = REPO / "models" / "superres" / "textzoom_stroke_restorer.pt"
IMAGE_SIZE = (128, 32)  # width, height used by the TextZoom paper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


class TextZoomSource:
    def __init__(self, folder: Path):
        self.folder = folder
        if not (folder / "data.mdb").exists():
            raise FileNotFoundError(folder / "data.mdb")
        probe = lmdb.open(str(folder), readonly=True, lock=False, readahead=False, max_readers=1)
        with probe.begin() as transaction:
            self.count = int(transaction.get(b"num-samples"))
        probe.close()
        self.env = None

    def transaction(self):
        # Open lazily inside each DataLoader process.
        if self.env is None:
            self.env = lmdb.open(
                str(self.folder), readonly=True, lock=False, readahead=False,
                meminit=False, max_readers=32,
            )
        return self.env.begin(write=False)


class TextZoomPairs(Dataset):
    def __init__(self, sources, keys):
        self.sources = sources
        self.keys = keys

    def __len__(self):
        return len(self.keys)

    @staticmethod
    def image(value):
        if value is None:
            raise KeyError("missing TextZoom image")
        pil = Image.open(BytesIO(value)).convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BICUBIC)
        array = np.asarray(pil, dtype=np.float32) / 255.0
        return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)

    def __getitem__(self, item):
        source_index, one_based_index = self.keys[item]
        suffix = f"{one_based_index:09d}".encode()
        with self.sources[source_index].transaction() as transaction:
            low = self.image(transaction.get(b"image_lr-" + suffix))
            high = self.image(transaction.get(b"image_hr-" + suffix))
        return low, high


def psnr_per_image(images, targets):
    mse = ((images - targets) ** 2).mean(dim=(1, 2, 3))
    return -10.0 * torch.log10(mse.clamp_min(1e-10))


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    input_scores, output_scores = [], []
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        input_scores.extend(psnr_per_image(inputs, targets).cpu().tolist())
        output_scores.extend(psnr_per_image(outputs, targets).cpu().tolist())
    return float(np.mean(input_scores)), float(np.mean(output_scores))


def main():
    args = parse_args()
    if args.smoke:
        args.epochs, args.val_size = 2, 64
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    sources = [TextZoomSource(args.data / "train1"), TextZoomSource(args.data / "train2")]
    keys = [(source_index, index)
            for source_index, source in enumerate(sources)
            for index in range(1, source.count + 1)]
    rng = random.Random(args.seed)
    rng.shuffle(keys)
    if args.smoke:
        keys = keys[:512]
    val_size = min(args.val_size, max(1, len(keys) // 5))
    val_keys, train_keys = keys[:val_size], keys[val_size:]
    print(f"TextZoom train pairs: {len(train_keys):,}; internal validation pairs: {len(val_keys):,}")
    print("Official TextZoom test subsets are untouched.")

    train_loader = DataLoader(
        TextZoomPairs(sources, train_keys), batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, pin_memory=False,
    )
    val_loader = DataLoader(
        TextZoomPairs(sources, val_keys), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=False,
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = StrokeRestorer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_gain, best_epoch, stale = -math.inf, 0, 0
    print("device:", device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            pixel = (outputs - targets).abs().mean()
            mask = torch.ones_like(inputs[:, :1])
            loss = pixel + 0.20 * gradient_loss(outputs, targets, mask)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

        input_psnr, output_psnr = validate(model, val_loader, device)
        gain = output_psnr - input_psnr
        scheduler.step(gain)
        improved = gain > best_gain + 0.01
        if improved:
            best_gain, best_epoch, stale = gain, epoch, 0
            torch.save({
                "model": model.state_dict(),
                "architecture": "StrokeRestorer",
                "preprocessing": "TextZoom RGB resized to 128x32 with bicubic",
                "training_sources": [str(source.folder) for source in sources],
                "train_pairs": len(train_keys),
                "val_pairs": len(val_keys),
                "epoch": epoch,
                "val_input_psnr": input_psnr,
                "val_output_psnr": output_psnr,
                "val_gain": gain,
                "image_size": IMAGE_SIZE,
                "seed": args.seed,
            }, args.output)
        else:
            stale += 1
        marker = " * best" if improved else ""
        print(
            f"epoch {epoch:02d}: train={np.mean(losses):.4f} "
            f"val input={input_psnr:.2f}dB output={output_psnr:.2f}dB "
            f"gain={gain:+.2f}dB{marker}"
        )
        if stale >= args.patience:
            print(f"early stopping after {args.patience} epochs without improvement")
            break

    print(f"saved best epoch {best_epoch}, validation gain {best_gain:+.2f}dB")
    print("checkpoint:", args.output)


if __name__ == "__main__":
    main()
