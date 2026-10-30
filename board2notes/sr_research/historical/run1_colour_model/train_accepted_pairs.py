#!/usr/bin/env python3
"""Train a small stroke-preserving restoration CNN on accepted paired crops.

The paired inputs have already been aligned to the clear target canvas. At inference time a distant
crop is likewise enlarged to the desired output size first; this network then restores strokes at
that size.

Changes after the 17 Oct review (see the notebook, section 6):
  1. Fresh random patches every epoch. Patch choice was seeded by the sample index only, so every
     epoch re-used the same 1,200 patches.
  2. Board normalisation (pair_data.normalise_board) on input and target independently, so the loss
     measures strokes rather than the exposure / white-balance difference between two photos.
     The SAME normalisation must be applied to a new photo before the model (see restore_full).
  3. Patches are centred on ink 75% of the time and never on warp fill; fill pixels are left out of
     the loss. Validation uses 8 fixed ink-centred patches per pair, real pairs only, and reports the
     gain over the untouched input (the model starts as an identity map, so the gain is what matters).
  4. Whole input photos are held out for validation and test (pair_data.assign_photo_splits).
  5. Optional --min-spp to train only on pairs that needed at most a given enlargement.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pair_data  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO / "outputs" / "paired_dataset_17oct" / "accepted_manifest.csv"
DEFAULT_MODEL = REPO / "models" / "superres" / "accepted_pair_restorer.pt"
NORMALISATION = "pair_data.normalise_board v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8, help="patches per step (multiple of --patches-per-pair)")
    parser.add_argument("--patches-per-pair", type=int, default=4, help="patches cut from each loaded pair")
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-spp", type=float, default=0.0,
                        help="train only on real pairs with source_pixels_per_target_pixel >= this (0 = all)")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--smoke", action="store_true", help="2 epochs x 3 steps: check that everything runs")
    return parser.parse_args()


def read_accepted(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    wrong = [row["pair_id"] for row in rows if row["review_status"] != "accepted"]
    if wrong:
        raise ValueError(f"Manifest contains {len(wrong)} non-accepted rows")
    return rows


def save_split_manifest(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_pair(row: dict[str, str], root: Path = REPO):
    """Normalised input, normalised target (float32 RGB in [0,1]) and the mask of real pixels."""
    raw_in = read_rgb(root / row["input_path"])
    raw_tg = read_rgb(root / row["target_path"])
    if raw_in.shape != raw_tg.shape:
        raise ValueError(f"Canvas mismatch for {row['pair_id']}")
    ok = pair_data.valid_mask(raw_in) & pair_data.valid_mask(raw_tg)
    return pair_data.normalise_board(raw_in), pair_data.normalise_board(raw_tg), ok


def to_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()


class PairPatches(Dataset):
    """Training patches. Each item loads one pair and returns `per_pair` patches from it."""

    def __init__(self, rows, patch_size, length, seed, per_pair=4, ink_prob=0.75):
        self.rows, self.patch, self.length = rows, patch_size, length
        self.seed, self.per_pair, self.ink_prob = seed, per_pair, ink_prob
        self.epoch = 0                                   # set by the training loop -> new patches each epoch

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        rng = random.Random(self.seed + self.epoch * 10_000_019 + index * 1_000_003)
        row = self.rows[rng.randrange(len(self.rows))]
        inp, tgt, ok = load_pair(row)
        ink = pair_data.ink_mask(tgt, grow=0) & ok
        xs, ys, ms = [], [], []
        for _ in range(self.per_pair):
            y, x, size = pair_data.sample_patch(rng, inp, tgt, self.patch, ok, ink, self.ink_prob)
            a, b = inp[y:y + size, x:x + size], tgt[y:y + size, x:x + size]
            m = ok[y:y + size, x:x + size].astype(np.float32)[..., None]
            if size != self.patch:                       # canvas smaller than a patch
                shape = (self.patch, self.patch)
                a = cv2.resize(a, shape, interpolation=cv2.INTER_CUBIC)
                b = cv2.resize(b, shape, interpolation=cv2.INTER_CUBIC)
                m = cv2.resize(m, shape, interpolation=cv2.INTER_NEAREST)[..., None]
            if rng.random() < 0.5:
                a, b, m = a[:, ::-1], b[:, ::-1], m[:, ::-1]
            xs.append(to_tensor(a)); ys.append(to_tensor(b)); ms.append(to_tensor(m))
        return torch.stack(xs), torch.stack(ys), torch.stack(ms)


def fixed_val_patches(rows, patch_size, per_pair=8):
    """Load each validation pair once; keep 8 fixed ink-centred patches (input, target, mask)."""
    xs, ys, ms = [], [], []
    for row in rows:
        inp, tgt, ok = load_pair(row)
        for y, x, size in pair_data.fixed_patches(row["pair_id"], tgt, ok, patch_size, per_pair):
            a, b = inp[y:y + size, x:x + size], tgt[y:y + size, x:x + size]
            m = ok[y:y + size, x:x + size].astype(np.float32)[..., None]
            if size != patch_size:
                continue
            xs.append(to_tensor(a)); ys.append(to_tensor(b)); ms.append(to_tensor(m))
    return torch.stack(xs), torch.stack(ys), torch.stack(ms)


class StrokeRestorer(nn.Module):
    """SRCNN-style residual network: predicts corrections, not new content."""

    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 64, 9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 5, padding=2),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.clamp(image + self.body(image), 0.0, 1.0)


def masked_l1(prediction, target, mask):
    return ((prediction - target).abs() * mask).sum() / (mask.sum() * prediction.shape[1]).clamp_min(1.0)


def gradient_loss(prediction, target, mask):
    mx = mask[..., :, 1:] * mask[..., :, :-1]
    my = mask[..., 1:, :] * mask[..., :-1, :]
    dx = ((prediction[..., :, 1:] - prediction[..., :, :-1]) - (target[..., :, 1:] - target[..., :, :-1])).abs()
    dy = ((prediction[..., 1:, :] - prediction[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :])).abs()
    c = prediction.shape[1]
    return (dx * mx).sum() / (mx.sum() * c).clamp_min(1.0) + (dy * my).sum() / (my.sum() * c).clamp_min(1.0)


def batch_psnr(images, targets, masks):
    err = ((images - targets) ** 2 * masks).sum(dim=(1, 2, 3)) / (masks.sum(dim=(1, 2, 3)) * 3).clamp_min(1.0)
    return (-10.0 * torch.log10(err.clamp_min(1e-10))).tolist()


@torch.no_grad()
def validate(model, val, device, batch_size=16):
    model.eval()
    xs, ys, ms = val
    out_psnr, in_psnr = [], []
    for i in range(0, len(xs), batch_size):
        x, y, m = xs[i:i + batch_size].to(device), ys[i:i + batch_size].to(device), ms[i:i + batch_size].to(device)
        out_psnr += batch_psnr(model(x), y, m)
        in_psnr += batch_psnr(x, y, m)
    return float(np.mean(out_psnr)), float(np.mean(in_psnr))


# ------------------------------------------------------------------------------------------------
# Whole-canvas restoration and evaluation (used by the notebook, section 6a-6c)
# ------------------------------------------------------------------------------------------------
@torch.no_grad()
def restore_full(model, norm_rgb: np.ndarray, device, tile: int = 256, context: int = 24) -> np.ndarray:
    """Restore a whole NORMALISED canvas (float RGB in [0,1]) tile by tile, at its native scale."""
    model.eval()
    h, w = norm_rgb.shape[:2]
    padded = np.pad(norm_rgb, ((context, context), (context, context), (0, 0)), mode="reflect")
    output = np.empty_like(norm_rgb)
    for y in range(0, h, tile):
        th = min(tile, h - y)
        for x in range(0, w, tile):
            tw = min(tile, w - x)
            patch = padded[y:y + th + 2 * context, x:x + tw + 2 * context]
            pred = model(to_tensor(patch).unsqueeze(0).to(device))[0].permute(1, 2, 0).cpu().numpy()
            output[y:y + th, x:x + tw] = pred[context:context + th, context:context + tw]
    return output


def evaluate_pairs(model, rows, device, root: Path = REPO) -> list[dict]:
    """Whole-canvas PSNR (normalised space) of the untouched input and of the restored output against
    the clear target, over all real pixels and over the stroke area only."""
    results = []
    for row in rows:
        inp, tgt, ok = load_pair(row, root)
        out = restore_full(model, inp, device)
        ink = pair_data.ink_mask(tgt, grow=3) & ok
        results.append({
            "pair_id": row["pair_id"], "group_id": row.get("group_id", ""), "split": row.get("split", ""),
            "input_role": row.get("input_role", ""),
            "spp_band": pair_data.spp_band(row.get("source_pixels_per_target_pixel")),
            "psnr_input_all": pair_data.psnr(inp, tgt, ok), "psnr_restored_all": pair_data.psnr(out, tgt, ok),
            "psnr_input_ink": pair_data.psnr(inp, tgt, ink), "psnr_restored_ink": pair_data.psnr(out, tgt, ink),
        })
        r = results[-1]
        r["gain_all"] = r["psnr_restored_all"] - r["psnr_input_all"]
        r["gain_ink"] = r["psnr_restored_ink"] - r["psnr_input_ink"]
    return results


def load_model(path: Path, device) -> StrokeRestorer:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("normalisation") != NORMALISATION:
        raise ValueError(f"{path} was not trained with {NORMALISATION}; retrain with this script")
    model = StrokeRestorer()
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs, args.steps_per_epoch = 2, 3
    if args.batch_size % args.patches_per_pair:
        raise ValueError("--batch-size must be a multiple of --patches-per-pair")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    rows = pair_data.assign_photo_splits(read_accepted(args.manifest))
    split_path = args.manifest.with_name("accepted_split_manifest.csv")
    save_split_manifest(rows, split_path)
    train_rows = [r for r in rows if r["split"] == "train"
                  and (r["pair_type"] != "real" or float(r["source_pixels_per_target_pixel"] or 0) >= args.min_spp)]
    val_rows = [r for r in rows if r["split"] == "val"]
    for name in ("train", "val", "test"):
        part = [r for r in rows if r["split"] == name]
        real = sum(r["pair_type"] == "real" for r in part)
        photos = sorted({r["input_photo"] for r in part if r["pair_type"] == "real"})
        print(f"{name:>5}: {len(part):>3} pairs ({real} real, {len(part) - real} synthetic) "
              f"from {len(photos)} input photos")
    if args.min_spp:
        print(f"--min-spp {args.min_spp}: training on {len(train_rows)} pairs")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    items_per_step = args.batch_size // args.patches_per_pair
    train_data = PairPatches(train_rows, args.patch_size, args.steps_per_epoch * items_per_step,
                             args.seed, per_pair=args.patches_per_pair)
    train_loader = DataLoader(train_data, batch_size=items_per_step, shuffle=False, num_workers=args.workers)
    print("loading fixed validation patches ...")
    val = fixed_val_patches(val_rows if not args.smoke else val_rows[:2], args.patch_size)

    model = StrokeRestorer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    _, val_input_psnr = validate(model, val, device)
    print(f"device: {device}; {len(val[0])} validation patches; untouched input PSNR {val_input_psnr:.2f} dB")
    best_psnr, best_epoch, bad_epochs = -math.inf, 0, 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_data.epoch = epoch                          # fresh patches every epoch
        model.train()
        train_losses = []
        for inputs, targets, masks in train_loader:
            inputs, targets, masks = (t.flatten(0, 1).to(device) for t in (inputs, targets, masks))
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = masked_l1(outputs, targets, masks) + 0.20 * gradient_loss(outputs, targets, masks)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach()))

        val_psnr, _ = validate(model, val, device)
        scheduler.step(val_psnr)
        improved = val_psnr > best_psnr + 0.01
        if improved:
            best_psnr, best_epoch, bad_epochs = val_psnr, epoch, 0
            torch.save({
                "model": model.state_dict(), "architecture": "StrokeRestorer", "normalisation": NORMALISATION,
                "epoch": epoch, "val_psnr": val_psnr, "val_input_psnr": val_input_psnr,
                "patch_size": args.patch_size, "min_spp": args.min_spp,
                "manifest": str(args.manifest), "split_manifest": str(split_path),
            }, args.output)
        else:
            bad_epochs += 1
        print(f"epoch {epoch:02d}: train={np.mean(train_losses):.4f} val_PSNR={val_psnr:.2f}dB "
              f"(gain over input {val_psnr - val_input_psnr:+.2f}dB){' * best' if improved else ''}")
        if bad_epochs >= args.patience:
            print(f"early stopping after {args.patience} epochs without improvement")
            break

    print(f"saved best epoch {best_epoch} ({best_psnr:.2f}dB, gain {best_psnr - val_input_psnr:+.2f}dB) "
          f"to {args.output}")
    print(f"saved photo-held-out splits to {split_path}")


if __name__ == "__main__":
    main()
