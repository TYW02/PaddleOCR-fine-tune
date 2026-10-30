#!/usr/bin/env python3
"""Train a deeper, stroke-aware residual CNN on TextZoom.

Experiment changes relative to train_textzoom_restorer.py:
  * six residual blocks instead of three convolution layers;
  * edge/stroke-weighted L1 plus gradient loss;
  * the same 128x32 geometry for training and validation;
  * checkpoint selection using OCR CER on a fixed validation subset.

The official TextZoom test folders are never opened by this script.
"""

from __future__ import annotations

import argparse
import math
import os
import random
from io import BytesIO
from pathlib import Path

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import cv2
import jiwer
import lmdb
import numpy as np
import torch
from PIL import Image
from paddleocr import TextRecognition
from torch import nn
from torch.utils.data import DataLoader, Dataset


REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO / "data" / "textzoom"
DEFAULT_OUTPUT = REPO / "models" / "superres" / "textzoom_residual_stroke.pt"
IMAGE_SIZE = (128, 32)  # width, height


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--ocr-val-size", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--stroke-weight", type=float, default=4.0)
    parser.add_argument("--gradient-weight", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


class TextZoomSource:
    def __init__(self, folder):
        self.folder = Path(folder)
        if not (self.folder / "data.mdb").exists():
            raise FileNotFoundError(self.folder / "data.mdb")
        probe = lmdb.open(str(self.folder), readonly=True, lock=False, readahead=False, max_readers=1)
        with probe.begin() as transaction:
            self.count = int(transaction.get(b"num-samples"))
        probe.close()
        self.env = None

    def transaction(self):
        if self.env is None:
            self.env = lmdb.open(
                str(self.folder), readonly=True, lock=False, readahead=False,
                meminit=False, max_readers=32,
            )
        return self.env.begin(write=False)


class TextZoomPairs(Dataset):
    def __init__(self, sources, keys, include_label=False):
        self.sources, self.keys, self.include_label = sources, keys, include_label

    def __len__(self):
        return len(self.keys)

    @staticmethod
    def image(value):
        pil = Image.open(BytesIO(value)).convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BICUBIC)
        array = np.asarray(pil, dtype=np.float32) / 255.0
        return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)

    def __getitem__(self, item):
        source_index, one_based_index = self.keys[item]
        suffix = f"{one_based_index:09d}".encode()
        with self.sources[source_index].transaction() as transaction:
            low = self.image(transaction.get(b"image_lr-" + suffix))
            high = self.image(transaction.get(b"image_hr-" + suffix))
            label = transaction.get(b"label-" + suffix).decode("utf-8")
        return (low, high, label) if self.include_label else (low, high)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, features):
        return features + 0.1 * self.body(features)


class ResidualStrokeRestorer(nn.Module):
    """Predict a conservative RGB correction while using deeper text features."""

    def __init__(self, blocks=6, channels=64):
        super().__init__()
        self.entry = nn.Conv2d(3, channels, 5, padding=2)
        self.blocks = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.exit = nn.Conv2d(channels, 3, 3, padding=1)
        nn.init.zeros_(self.exit.weight)
        nn.init.zeros_(self.exit.bias)

    def forward(self, image):
        features = torch.relu(self.entry(image))
        correction = self.exit(self.blocks(features))
        return torch.clamp(image + correction, 0.0, 1.0)


def edge_weights(target, strength):
    grey = target.mean(dim=1, keepdim=True)
    dx = torch.nn.functional.pad((grey[..., :, 1:] - grey[..., :, :-1]).abs(), (0, 1, 0, 0))
    dy = torch.nn.functional.pad((grey[..., 1:, :] - grey[..., :-1, :]).abs(), (0, 0, 0, 1))
    magnitude = torch.sqrt(dx.square() + dy.square() + 1e-8)
    scale = magnitude.mean(dim=(2, 3), keepdim=True).clamp_min(1e-4)
    emphasis = torch.clamp(magnitude / (3.0 * scale), 0.0, 1.0)
    return 1.0 + strength * emphasis


def stroke_weighted_l1(prediction, target, strength):
    weights = edge_weights(target, strength)
    return ((prediction - target).abs() * weights).sum() / (weights.sum() * prediction.shape[1])


def gradient_loss(prediction, target):
    dx = ((prediction[..., :, 1:] - prediction[..., :, :-1])
          - (target[..., :, 1:] - target[..., :, :-1])).abs().mean()
    dy = ((prediction[..., 1:, :] - prediction[..., :-1, :])
          - (target[..., 1:, :] - target[..., :-1, :])).abs().mean()
    return dx + dy


def psnr_per_image(images, targets):
    mse = ((images - targets) ** 2).mean(dim=(1, 2, 3))
    return -10.0 * torch.log10(mse.clamp_min(1e-10))


@torch.no_grad()
def validate_psnr(model, loader, device):
    model.eval()
    before, after = [], []
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        before.extend(psnr_per_image(inputs, targets).cpu().tolist())
        after.extend(psnr_per_image(outputs, targets).cpu().tolist())
    return float(np.mean(before)), float(np.mean(after))


def clean_text(text):
    return " ".join(str(text).lower().split())


def uint8_bgr(tensor):
    array = tensor.detach().permute(1, 2, 0).cpu().numpy()
    rgb = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


@torch.no_grad()
def validate_ocr(model, loader, recogniser, device):
    model.eval()
    references, images = [], []
    for inputs, _, labels in loader:
        outputs = model(inputs.to(device))
        references.extend(labels)
        images.extend(uint8_bgr(output) for output in outputs)
    predictions = recogniser.predict(images)
    predicted_text = [str(result["rec_text"]) for result in predictions]
    cers = [min(float(jiwer.cer(clean_text(ref), clean_text(pred) or " ")), 1.0)
            for ref, pred in zip(references, predicted_text)]
    exact = [clean_text(ref) == clean_text(pred) for ref, pred in zip(references, predicted_text)]
    return float(np.mean(cers)), float(np.mean(exact))


def main():
    args = parse_args()
    if args.smoke:
        args.epochs, args.val_size, args.ocr_val_size = 2, 64, 16
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    sources = [TextZoomSource(args.data / "train1"), TextZoomSource(args.data / "train2")]
    keys = [(source_index, index)
            for source_index, source in enumerate(sources)
            for index in range(1, source.count + 1)]
    random.Random(args.seed).shuffle(keys)
    if args.smoke:
        keys = keys[:512]
    val_size = min(args.val_size, max(1, len(keys) // 5))
    val_keys, train_keys = keys[:val_size], keys[val_size:]
    ocr_keys = val_keys[:min(args.ocr_val_size, len(val_keys))]
    print(f"TextZoom train pairs: {len(train_keys):,}; validation: {len(val_keys):,}; OCR validation: {len(ocr_keys):,}")
    print("Official TextZoom test subsets are untouched.")

    train_loader = DataLoader(TextZoomPairs(sources, train_keys), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(TextZoomPairs(sources, val_keys), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers)
    ocr_loader = DataLoader(TextZoomPairs(sources, ocr_keys, include_label=True),
                            batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ResidualStrokeRestorer(args.blocks, args.channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    recogniser = TextRecognition(model_name="PP-OCRv6_medium_rec")

    baseline_cer, baseline_exact = validate_ocr(nn.Identity().to(device), ocr_loader, recogniser, device)
    print(f"device: {device}; baseline OCR CER={baseline_cer:.4f}, exact={baseline_exact:.4f}")
    print(f"architecture: {args.blocks} residual blocks, {args.channels} channels")
    print(f"loss: edge-weighted L1 (strength {args.stroke_weight}) + {args.gradient_weight} x gradient")

    best_key, best_epoch, stale = (math.inf, 0.0, -math.inf), 0, 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = (stroke_weighted_l1(outputs, targets, args.stroke_weight)
                    + args.gradient_weight * gradient_loss(outputs, targets))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

        input_psnr, output_psnr = validate_psnr(model, val_loader, device)
        ocr_cer, ocr_exact = validate_ocr(model, ocr_loader, recogniser, device)
        gain = output_psnr - input_psnr
        key = (ocr_cer, -ocr_exact, -gain)
        improved = key < best_key
        if improved:
            best_key, best_epoch, stale = key, epoch, 0
            torch.save({
                "model": model.state_dict(), "architecture": "ResidualStrokeRestorer",
                "blocks": args.blocks, "channels": args.channels,
                "preprocessing": "TextZoom RGB resized to 128x32 with bicubic",
                "loss": f"edge-weighted L1 strength={args.stroke_weight} + {args.gradient_weight}*gradient",
                "train_pairs": len(train_keys), "val_pairs": len(val_keys),
                "ocr_val_pairs": len(ocr_keys), "epoch": epoch,
                "val_input_psnr": input_psnr, "val_output_psnr": output_psnr,
                "val_gain": gain, "val_ocr_cer": ocr_cer, "val_ocr_exact": ocr_exact,
                "baseline_ocr_cer": baseline_cer, "baseline_ocr_exact": baseline_exact,
                "image_size": IMAGE_SIZE, "seed": args.seed,
            }, args.output)
        else:
            stale += 1
        marker = " * best OCR" if improved else ""
        print(f"epoch {epoch:02d}: train={np.mean(losses):.4f} PSNR gain={gain:+.2f}dB "
              f"OCR CER={ocr_cer:.4f} exact={ocr_exact:.4f}{marker}")
        if stale >= args.patience:
            print(f"early stopping after {args.patience} epochs without OCR improvement")
            break

    print(f"saved best OCR checkpoint from epoch {best_epoch}: {args.output}")


if __name__ == "__main__":
    main()
