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
  6. Greyscale (run 2). Run 1 in colour mostly learned to recolour the 17 Oct blue marker (dark and dull
     from far away, bright blue close up): +0.9 dB on that board, -0.2 dB on black-marker boards.
     Input and target are now normalised, then converted to grey (same value in all 3 channels).
     The SAME two steps must be applied to a new photo before the model; to give PaddleOCR the
     result in natural tones, multiply back with pair_data.denormalise_grey.
  7. Ink-depth matching (run 3). Run 2 (grey) still learned to LIGHTEN the 17 Oct strokes (the marker
     looks darker from far away). The target's ink depth is now matched to the input's
     (pair_data.match_ink_depth), so only stroke shape and sharpness are left to learn. Validation and
     test PSNR are measured against this matched target too.
  8. --extra-manifest: add training pairs from other boards (make_airdrop_synthetic_pairs.py builds
     synthetic pairs from sharp Airdrop crops of boards that are NOT test boards).
  9. Canvases smaller than a patch are padded (and the padding masked out of the loss) instead of being
     stretched, which changed the stroke scale.
 10. --extra-val-manifest / --extra-val-groups (run 4): add real pairs from another board to validation.
     The 17 Oct validation photos are already nearly sharp, so on their own they could not show any
     sharpening and early stopping kept an almost untrained model (run 3: epoch 1). Each validation set
     counts EQUALLY when choosing the best epoch (mean of the per-set gains over the untouched input).
 11. Stroke-aware loss (M1). Run 4 raised PSNR on an unseen board (+1.36 dB) without lowering PaddleOCR's
     CER: a pixel loss is dominated by board pixels and rewards an averaged "soft blob" where a stroke is
     uncertain. Three optional terms, all OFF by default so run 4 is reproduced exactly:
       --ink-weight W     pixels on and next to the target's strokes count W times in the L1 term
       --dice-weight D    soft Dice between predicted and target ink: penalises missing or extra ink
       --cldice-weight C  soft clDice (Shit et al., CVPR 2021): compares stroke SKELETONS, so a stroke
                          that is broken, or two strokes merged into one blob, is penalised even when
                          only a few pixels differ
     Validation now also reports the stroke Dice (thresholded ink overlap) of the output and of the
     untouched input; --select-on chooses whether the best epoch is picked on PSNR gain (default, as
     in run 4) or on stroke-Dice gain.
     `python train_accepted_pairs.py --self-test` checks the loss functions on synthetic strokes.
 12. Recognition loss (M2). M1 (stroke-aware pixel terms) did not change PaddleOCR's reading either: the
     model barely reduced the stroke terms, and a shape loss does not know what a LETTER is. Every
     text-SR method that improves reading ties its loss to a recogniser (TextSR, TPGSR, TATT, Text
     Gestalt). --recognizer loads a frozen PyTorch CRNN (scripts/line_recognizer.py, trained by
     scripts/train_line_recognizer.py on IAM + our whiteboard crops, never on the B2/B3 boards); each
     step, --rec-lines-per-step WHOLE labelled lines are restored and read by it, and its CTC loss
     against the line's label is added to the pixel loss. Only lines whose canvas is at most
     --rec-max-height px tall are used (the Airdrop-scale lines; the 17 Oct canvases are far bigger).
     --rec-ratio sets the term's size: its weight is chosen at the start so that rec term = ratio x
     pixel loss (measured on the untouched inputs). Validation also reports the recogniser's CER on
     the labelled validation lines (input vs output). PaddleOCR, a different recogniser, still judges.
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
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import line_recognizer  # noqa: E402
import pair_data  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO / "outputs" / "paired_dataset_17oct" / "accepted_manifest.csv"
DEFAULT_MODEL = REPO / "models" / "superres" / "accepted_pair_restorer.pt"
NORMALISATION = "pair_data.normalise_board v1 + to_grey3 + match_ink_depth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--extra-manifest", type=Path, nargs="*", default=[],
                        help="extra TRAINING pairs (e.g. synthetic pairs from other boards); never used for val/test")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8, help="patches per step (multiple of --patches-per-pair)")
    parser.add_argument("--patches-per-pair", type=int, default=4, help="patches cut from each loaded pair")
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--extra-val-manifest", type=Path, default=None,
                        help="manifest with extra REAL validation pairs (e.g. paired_dataset_airdrop/airdrop_test_manifest.csv)")
    parser.add_argument("--extra-val-groups", nargs="*", default=[],
                        help="group_ids from --extra-val-manifest to use for validation (e.g. airdrop_B2); "
                             "the remaining groups stay test-only")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-spp", type=float, default=0.0,
                        help="train only on real pairs with source_pixels_per_target_pixel >= this (0 = all)")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--smoke", action="store_true", help="2 epochs x 3 steps: check that everything runs")
    # stroke-aware loss (M1); the defaults reproduce run 4
    parser.add_argument("--ink-weight", type=float, default=1.0,
                        help="weight of stroke pixels (and a 2 px band around them) in the L1 term; 1 = plain L1")
    parser.add_argument("--dice-weight", type=float, default=0.0, help="weight of the soft ink Dice term")
    parser.add_argument("--cldice-weight", type=float, default=0.0, help="weight of the soft clDice (skeleton) term")
    parser.add_argument("--skeleton-iterations", type=int, default=15,
                        help="erosion steps of the soft skeleton; must be >= half the widest stroke in px")
    parser.add_argument("--select-on", choices=["psnr", "dice", "recog"], default="psnr",
                        help="validation score that picks the best epoch (gain over the untouched input); "
                             "recog = drop in the recogniser's CER on the labelled validation lines")
    # recognition loss (M2); off unless --recognizer and --rec-ratio are given
    parser.add_argument("--recognizer", type=Path, default=None, help="frozen line recogniser checkpoint")
    parser.add_argument("--rec-ratio", type=float, default=0.0,
                        help="size of the recognition term relative to the pixel loss at the start (0 = off)")
    parser.add_argument("--rec-lines-per-step", type=int, default=2)
    parser.add_argument("--rec-max-height", type=int, default=300,
                        help="only lines whose canvas is at most this tall (px) get the recognition term: "
                             "Airdrop lines are 80-281 px, 17 Oct canvases 456-2502 px")
    parser.add_argument("--self-test", action="store_true", help="check the loss functions and exit")
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
    """Normalised GREY input, normalised GREY target with its ink depth matched to the input (float32,
    grey in all 3 channels, [0,1]) and the mask of real (non-fill) pixels."""
    raw_in = read_rgb(root / row["input_path"])
    raw_tg = read_rgb(root / row["target_path"])
    if raw_in.shape != raw_tg.shape:
        raise ValueError(f"Canvas mismatch for {row['pair_id']}")
    ok = pair_data.valid_mask(raw_in) & pair_data.valid_mask(raw_tg)
    inp = pair_data.to_grey3(pair_data.normalise_board(raw_in))
    tgt = pair_data.to_grey3(pair_data.normalise_board(raw_tg))
    tgt = pair_data.match_ink_depth(inp, tgt, ok)
    return inp, tgt, ok


def cut_patch(inp, tgt, ok, y, x, size, patch):
    """Cut a size x size patch; if the canvas is smaller than a patch, pad to patch x patch with white
    board (1.0) and mask the padding out of the loss (stretching would change the stroke scale)."""
    a, b = inp[y:y + size, x:x + size], tgt[y:y + size, x:x + size]
    m = ok[y:y + size, x:x + size].astype(np.float32)[..., None]
    if size == patch:
        return a, b, m
    pa = np.ones((patch, patch, 3), np.float32); pb = pa.copy(); pm = np.zeros((patch, patch, 1), np.float32)
    pa[:size, :size], pb[:size, :size], pm[:size, :size] = a, b, m
    return pa, pb, pm


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
            a, b, m = cut_patch(inp, tgt, ok, y, x, size, self.patch)
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
            a, b, m = cut_patch(inp, tgt, ok, y, x, size, patch_size)
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


# ------------------------------------------------------------------------------------------------
# Stroke-aware loss terms (M1). Images are normalised grey (board ~1.0, ink darker), so "ink" is a
# pixel clearly below the board. INK_THRESHOLD is the same 0.75 used by pair_data.ink_mask.
# ------------------------------------------------------------------------------------------------
INK_THRESHOLD = 0.75
INK_RAMP = (0.60, 0.90)  # grey <= 0.60 is surely ink (1), >= 0.90 is surely board (0), linear in between


def soft_ink(image: torch.Tensor) -> torch.Tensor:
    """Differentiable ink "probability" per pixel, shape (B, 1, H, W). A clamped ramp rather than a
    sigmoid: a sigmoid gives every board pixel a small non-zero value, which adds up over a patch and
    spoils the Dice overlap; with the ramp the board is exactly 0."""
    lo, hi = INK_RAMP
    return ((hi - image[:, :1]) / (hi - lo)).clamp(0.0, 1.0)


def ink_weights(target: torch.Tensor, mask: torch.Tensor, ink_weight: float, grow: int = 2) -> torch.Tensor:
    """Per-pixel L1 weights: 1 on the board, ink_weight on the target's strokes and a `grow` px band
    around them (a blurred stroke spreads into that band). ink_weight = 1 returns the mask unchanged."""
    if ink_weight == 1.0:
        return mask
    ink = (target[:, :1] < INK_THRESHOLD).float()
    ink = F.max_pool2d(ink, 2 * grow + 1, stride=1, padding=grow)
    return mask * (1.0 + (ink_weight - 1.0) * ink)


def weighted_l1(prediction, target, weights):
    """Same as masked_l1 when weights == mask."""
    return ((prediction - target).abs() * weights).sum() / (weights.sum() * prediction.shape[1]).clamp_min(1.0)


def _per_patch_mean(values: torch.Tensor, target_ink: torch.Tensor, min_ink: float = 10.0) -> torch.Tensor:
    """Average over the patches whose target actually contains strokes (a blank patch has no shape)."""
    keep = target_ink > min_ink
    return values[keep].mean() if keep.any() else values.sum() * 0.0


def soft_dice_loss(prediction, target, mask, eps: float = 1.0) -> torch.Tensor:
    """1 - Dice overlap of predicted and target ink: high when strokes are missing or extra."""
    p, q = soft_ink(prediction) * mask, soft_ink(target) * mask
    inter = (p * q).sum(dim=(1, 2, 3))
    total_q = q.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + eps) / (p.sum(dim=(1, 2, 3)) + total_q + eps)
    return _per_patch_mean(1.0 - dice, total_q)


def soft_erode(image: torch.Tensor) -> torch.Tensor:
    """Grey erosion with a plus-shaped 3x3 element (min over the 4-neighbourhood)."""
    p1 = -F.max_pool2d(-image, (3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-image, (1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def soft_dilate(image: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(image, 3, stride=1, padding=1)


def soft_open(image: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(image))


def soft_skeleton(image: torch.Tensor, iterations: int) -> torch.Tensor:
    """Differentiable skeleton of an ink-probability map (Shit et al., clDice, CVPR 2021): what is lost by
    an opening at each erosion depth, accumulated. Thin strokes survive as their centre line."""
    skeleton = F.relu(image - soft_open(image))
    for _ in range(iterations):
        image = soft_erode(image)
        delta = F.relu(image - soft_open(image))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss(prediction, target, mask, iterations: int = 15, eps: float = 1.0) -> torch.Tensor:
    """1 - clDice. Topology precision: how much of the predicted skeleton lies on target ink.
    Topology sensitivity: how much of the target skeleton is covered by predicted ink. A gap in a stroke
    removes part of the target skeleton's coverage; a merged blob adds skeleton where there is no ink."""
    p, q = soft_ink(prediction) * mask, soft_ink(target) * mask
    sp, sq = soft_skeleton(p, iterations), soft_skeleton(q, iterations)
    precision = ((sp * q).sum(dim=(1, 2, 3)) + eps) / (sp.sum(dim=(1, 2, 3)) + eps)
    sensitivity = ((sq * p).sum(dim=(1, 2, 3)) + eps) / (sq.sum(dim=(1, 2, 3)) + eps)
    cldice = 2.0 * precision * sensitivity / (precision + sensitivity)
    return _per_patch_mean(1.0 - cldice, q.sum(dim=(1, 2, 3)))


def training_loss(outputs, targets, masks, args) -> tuple[torch.Tensor, dict]:
    """Run 4 loss (L1 + 0.2 x gradient) plus the optional stroke terms. Returns the total and its parts."""
    weights = ink_weights(targets, masks, args.ink_weight)
    parts = {"l1": weighted_l1(outputs, targets, weights), "grad": 0.20 * gradient_loss(outputs, targets, masks)}
    if args.dice_weight:
        parts["dice"] = args.dice_weight * soft_dice_loss(outputs, targets, masks)
    if args.cldice_weight:
        parts["cldice"] = args.cldice_weight * soft_cldice_loss(outputs, targets, masks, args.skeleton_iterations)
    return sum(parts.values()), {k: float(v.detach()) for k, v in parts.items()}


# ------------------------------------------------------------------------------------------------
# Recognition loss (M2)
# ------------------------------------------------------------------------------------------------
def recognition_lines(rows, charset, max_height):
    """Labelled whole lines for the recognition term: (normalised grey input (H, W) tensor, encoded label,
    text, pair_id). Canvases taller than max_height and unlabelled rows are skipped."""
    lines, too_big, unlabelled = [], 0, 0
    for row in rows:
        text = (row.get("transcription_verified") or "").strip()
        if not text:
            unlabelled += 1
            continue
        header = cv2.imread(str(REPO / row["input_path"]), cv2.IMREAD_GRAYSCALE)
        if header is None or header.shape[0] > max_height:
            too_big += 1
            continue
        label = charset.encode(text)
        if not label:
            unlabelled += 1
            continue
        inp, _, _ = load_pair(row)
        lines.append((torch.from_numpy(np.ascontiguousarray(inp[..., 0])), label, text, row["pair_id"]))
    return lines, too_big, unlabelled


def recognition_loss(model, recognizer, items, device) -> torch.Tensor:
    """Mean CTC loss of the frozen recogniser reading the RESTORED lines (gradients reach the SR model)."""
    losses = []
    for grey, label, _, _ in items:
        restored = model(grey[None, None].repeat(1, 3, 1, 1).to(device))
        log_probs = recognizer(line_recognizer.ink_from_sr_output(restored))
        losses.append(line_recognizer.ctc_loss(log_probs, torch.tensor([log_probs.shape[0]]), [label]))
    return torch.stack(losses).mean().to(device)


@torch.no_grad()
def recognition_cer(model, recognizer, charset, items, device) -> tuple[float, float]:
    """Mean recogniser CER on (untouched input, restored output) of the labelled lines."""
    model.eval()
    before, after = [], []
    for grey, _, text, _ in items:
        x = grey[None, None].repeat(1, 3, 1, 1).to(device)
        for image, store in ((x, before), (model(x), after)):
            ink = line_recognizer.ink_from_sr_output(image)
            store.append(line_recognizer.cer(text, line_recognizer.read(
                recognizer, charset, ink, torch.tensor([ink.shape[3] // line_recognizer.DOWNSAMPLE]))[0]))
    return float(np.mean(before)), float(np.mean(after))


def batch_psnr(images, targets, masks):
    err = ((images - targets) ** 2 * masks).sum(dim=(1, 2, 3)) / (masks.sum(dim=(1, 2, 3)) * 3).clamp_min(1.0)
    return (-10.0 * torch.log10(err.clamp_min(1e-10))).tolist()


def batch_stroke_dice(images, targets, masks):
    """Hard (thresholded) ink Dice per patch; NaN for patches without target ink."""
    a = (images[:, :1] < INK_THRESHOLD).float() * masks
    b = (targets[:, :1] < INK_THRESHOLD).float() * masks
    inter, total = (a * b).sum(dim=(1, 2, 3)), a.sum(dim=(1, 2, 3)) + b.sum(dim=(1, 2, 3))
    dice = 2.0 * inter / total.clamp_min(1.0)
    dice[b.sum(dim=(1, 2, 3)) < 10] = float("nan")
    return dice.tolist()


@torch.no_grad()
def validate(model, val, device, batch_size=16):
    """Mean PSNR and mean stroke Dice of the model output and of the untouched input."""
    model.eval()
    xs, ys, ms = val
    out_psnr, in_psnr, out_dice, in_dice = [], [], [], []
    for i in range(0, len(xs), batch_size):
        x, y, m = xs[i:i + batch_size].to(device), ys[i:i + batch_size].to(device), ms[i:i + batch_size].to(device)
        out = model(x)
        out_psnr += batch_psnr(out, y, m)
        in_psnr += batch_psnr(x, y, m)
        out_dice += batch_stroke_dice(out, y, m)
        in_dice += batch_stroke_dice(x, y, m)
    return {"psnr": float(np.mean(out_psnr)), "input_psnr": float(np.mean(in_psnr)),
            "dice": float(np.nanmean(out_dice)), "input_dice": float(np.nanmean(in_dice))}


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
    for extra in args.extra_manifest:
        extra_rows = [{**r, "split": "train"} for r in read_accepted(extra)]
        train_rows += extra_rows
        print(f"extra training pairs: {len(extra_rows)} from {extra.name} "
              f"(groups: {sorted({r['group_id'] for r in extra_rows})})")
    val_rows = [r for r in rows if r["split"] == "val"]
    for name in ("train", "val", "test"):
        part = [r for r in rows if r["split"] == name]
        real = sum(r["pair_type"] == "real" for r in part)
        photos = sorted({r["input_photo"] for r in part if r["pair_type"] == "real"})
        print(f"{name:>5}: {len(part):>3} pairs ({real} real, {len(part) - real} synthetic) "
              f"from {len(photos)} input photos  [17 Oct manifest]")
    if args.min_spp:
        print(f"--min-spp {args.min_spp}: training on {len(train_rows)} pairs")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    items_per_step = args.batch_size // args.patches_per_pair
    train_data = PairPatches(train_rows, args.patch_size, args.steps_per_epoch * items_per_step,
                             args.seed, per_pair=args.patches_per_pair)
    train_loader = DataLoader(train_data, batch_size=items_per_step, shuffle=False, num_workers=args.workers)
    val_sets = {"17 Oct": val_rows}
    if args.extra_val_manifest is not None:
        with args.extra_val_manifest.open(newline="", encoding="utf-8") as handle:
            extra_val = [r for r in csv.DictReader(handle) if r["review_status"] == "accepted"]
        for group in args.extra_val_groups:
            chosen = [r for r in extra_val if r["group_id"] == group]
            if not chosen:
                raise ValueError(f"no accepted pairs with group_id {group} in {args.extra_val_manifest}")
            val_sets[group] = chosen
    print("loading fixed validation patches ...")
    val = {name: fixed_val_patches(part if not args.smoke else part[:2], args.patch_size)
           for name, part in val_sets.items()}

    model = StrokeRestorer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    start = {name: validate(model, v, device) for name, v in val.items()}   # identity model = untouched input
    print(f"device: {device}")

    # ---- recognition loss (M2) ----
    recognizer = charset = None
    rec_train, rec_val, rec_weight, rec_in_cer = [], [], 0.0, float("nan")
    if args.rec_ratio or args.select_on == "recog":
        if args.recognizer is None:
            raise ValueError("--rec-ratio / --select-on recog need --recognizer (run train_line_recognizer.py first)")
        recognizer, charset, rec_info = line_recognizer.load(args.recognizer, device)
        print(f"recogniser {args.recognizer.name}: epoch {rec_info.get('epoch')}, CER IAM val "
              f"{rec_info.get('iam_val_cer', float('nan')):.3f}, CNN board crops {rec_info.get('airdrop_b3_cer', float('nan')):.3f}")
        rec_train, too_big, unlabelled = recognition_lines(train_rows, charset, args.rec_max_height)
        rec_val, _, _ = recognition_lines([r for part in val_sets.values() for r in part], charset, args.rec_max_height)
        if args.smoke:
            rec_train, rec_val = rec_train[:20], rec_val[:2]
        groups = sorted({r["group_id"] for r in train_rows
                         if r["pair_id"] in {line[3] for line in rec_train}})
        print(f"recognition term: {len(rec_train)} labelled training lines (from {groups}); skipped "
              f"{too_big} canvases taller than {args.rec_max_height} px and {unlabelled} unlabelled; "
              f"{len(rec_val)} labelled validation lines")
        if args.rec_ratio and not rec_train:
            raise ValueError("no training line qualifies for the recognition term")
        if args.select_on == "recog" and not rec_val:
            raise ValueError("--select-on recog needs labelled validation lines no taller than --rec-max-height")
        rec_in_cer, _ = recognition_cer(model, recognizer, charset, rec_val, device) if rec_val else (float("nan"), 0)
    if args.rec_ratio:
        # weight chosen so that, on the untouched inputs, rec term = rec_ratio x pixel loss
        model.eval()
        pixel = []
        with torch.no_grad():
            for i, (inputs, targets, masks) in enumerate(train_loader):
                inputs, targets, masks = (t.flatten(0, 1).to(device) for t in (inputs, targets, masks))
                pixel.append(float(training_loss(model(inputs), targets, masks, args)[0]))
                if i >= 9:
                    break
            # a RANDOM sample of the recognition lines: the manifests are sorted by board, so the first
            # lines are not representative (M2 run 1 calibrated on B1/B4 lines only, and the term then ran
            # at 3-5x the pixel loss instead of 1x because the B5 lines are harder)
            calib = random.Random(args.seed + 3).sample(rec_train, min(60, len(rec_train)))
            rec_start = float(recognition_loss(model, recognizer, calib, device))
        rec_weight = args.rec_ratio * float(np.mean(pixel)) / max(rec_start, 1e-6)
        print(f"calibration on untouched inputs: pixel loss {np.mean(pixel):.4f}, recogniser CTC loss "
              f"{rec_start:.3f} -> rec weight {rec_weight:.5f} (rec term = {args.rec_ratio} x pixel loss)")
    rec_rng = random.Random(args.seed + 7)

    print(f"loss: L1 (ink weight {args.ink_weight}) + 0.2 x gradient"
          + (f" + {args.dice_weight} x Dice" if args.dice_weight else "")
          + (f" + {args.cldice_weight} x clDice ({args.skeleton_iterations} skeleton steps)" if args.cldice_weight else "")
          + (f" + {rec_weight:.5f} x recogniser CTC ({args.rec_lines_per_step} whole lines per step)" if rec_weight else "")
          + f"; best epoch chosen on {args.select_on} gain")
    for name, v in val.items():
        print(f"  validation set {name}: {len(val_sets[name])} pairs, {len(v[0])} patches, untouched input "
              f"PSNR {start[name]['input_psnr']:.2f} dB, stroke Dice {start[name]['input_dice']:.3f}")
    if rec_val:
        print(f"  recogniser CER on the {len(rec_val)} labelled validation lines, untouched input: {rec_in_cer:.3f}")
    val_input_psnr = float(np.mean([s["input_psnr"] for s in start.values()]))
    best_gain, best_epoch, bad_epochs = -math.inf, 0, 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    min_step = 0.01 if args.select_on == "psnr" else 0.001     # dB, or Dice / CER points

    for epoch in range(1, args.epochs + 1):
        train_data.epoch = epoch                          # fresh patches every epoch
        model.train()
        train_losses, parts_log = [], []
        for inputs, targets, masks in train_loader:
            inputs, targets, masks = (t.flatten(0, 1).to(device) for t in (inputs, targets, masks))
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss, parts = training_loss(outputs, targets, masks, args)
            if rec_weight:
                picks = rec_rng.sample(range(len(rec_train)), min(args.rec_lines_per_step, len(rec_train)))
                rec_term = rec_weight * recognition_loss(model, recognizer, [rec_train[i] for i in picks], device)
                loss = loss + rec_term
                parts["rec"] = float(rec_term.detach())
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach()))
            parts_log.append(parts)

        scores = {name: validate(model, v, device) for name, v in val.items()}
        psnr_gains = {name: s["psnr"] - s["input_psnr"] for name, s in scores.items()}
        dice_gains = {name: s["dice"] - s["input_dice"] for name, s in scores.items()}
        rec_out_cer = recognition_cer(model, recognizer, charset, rec_val, device)[1] if rec_val else float("nan")
        if args.select_on == "recog":
            gain = rec_in_cer - rec_out_cer                      # drop in recogniser CER
        else:
            chosen = psnr_gains if args.select_on == "psnr" else dice_gains
            gain = float(np.mean(list(chosen.values())))           # every validation set counts equally
        scheduler.step(gain)
        improved = gain > best_gain + min_step
        if improved:
            best_gain, best_epoch, bad_epochs = gain, epoch, 0
            mean_psnr_gain = float(np.mean(list(psnr_gains.values())))
            torch.save({
                "model": model.state_dict(), "architecture": "StrokeRestorer", "normalisation": NORMALISATION,
                "epoch": epoch, "val_psnr": val_input_psnr + mean_psnr_gain, "val_input_psnr": val_input_psnr,
                "val_sets": scores, "select_on": args.select_on,
                "loss": {"ink_weight": args.ink_weight, "dice_weight": args.dice_weight,
                         "cldice_weight": args.cldice_weight, "skeleton_iterations": args.skeleton_iterations,
                         "rec_ratio": args.rec_ratio, "rec_weight": rec_weight,
                         "recognizer": str(args.recognizer) if args.recognizer else None},
                "val_recog_cer": {"input": rec_in_cer, "restored": rec_out_cer, "lines": len(rec_val)},
                "patch_size": args.patch_size, "min_spp": args.min_spp,
                "extra_manifests": [str(p) for p in args.extra_manifest],
                "manifest": str(args.manifest), "split_manifest": str(split_path),
            }, args.output)
        else:
            bad_epochs += 1
        mean_parts = "  ".join(f"{k} {np.mean([p[k] for p in parts_log]):.4f}" for k in parts_log[0])
        per_set = "  ".join(f"{name} {psnr_gains[name]:+.2f}dB/{dice_gains[name]:+.3f}" for name in scores)
        recog = f" | recogniser CER {rec_in_cer:.3f}->{rec_out_cer:.3f}" if rec_val else ""
        print(f"epoch {epoch:02d}: train={np.mean(train_losses):.4f} ({mean_parts}) | val gain over input "
              f"(PSNR/stroke Dice): {per_set}{recog} | selection {gain:+.3f}{' * best' if improved else ''}")
        if bad_epochs >= args.patience:
            print(f"early stopping after {args.patience} epochs without improvement")
            break

    print(f"saved best epoch {best_epoch} (mean validation {args.select_on} gain {best_gain:+.3f}) to {args.output}")
    print(f"saved photo-held-out splits to {split_path}")


def self_test() -> None:
    """Checks the loss terms on synthetic strokes (no data needed). Run: --self-test."""
    torch.manual_seed(0)
    board = torch.ones(2, 3, 64, 64)
    target = board.clone()
    target[:, :, 20:24, 8:56] = 0.3                      # one horizontal stroke, 4 px wide
    target[:, :, 30:50, 30:34] = 0.3                     # one vertical stroke
    mask = torch.ones(2, 1, 64, 64)
    broken = target.clone(); broken[:, :, 20:24, 28:36] = 1.0          # 8 px gap in the stroke
    merged = target.clone(); merged[:, :, 24:30, 28:36] = 0.3          # strokes joined by a blob
    soft = F.avg_pool2d(F.pad(target, (2, 2, 2, 2), mode="replicate"), 5, stride=1)   # blurred copy
    checks = []
    checks.append(("identical -> Dice loss ~0", float(soft_dice_loss(target, target, mask)) < 0.02))
    checks.append(("identical -> clDice loss ~0", float(soft_cldice_loss(target, target, mask)) < 0.02))
    checks.append(("gap raises clDice", float(soft_cldice_loss(broken, target, mask)) > 0.03))
    checks.append(("merge raises clDice", float(soft_cldice_loss(merged, target, mask)) > 0.02))
    checks.append(("ink weight 1 == masked L1",
                   abs(float(weighted_l1(soft, target, ink_weights(target, mask, 1.0)))
                       - float(masked_l1(soft, target, mask))) < 1e-7))
    checks.append(("ink weight 5 > masked L1 for a blurred stroke",
                   float(weighted_l1(soft, target, ink_weights(target, mask, 5.0)))
                   > float(masked_l1(soft, target, mask))))
    blank = torch.ones(1, 3, 32, 32)
    checks.append(("blank patch -> no stroke loss", float(soft_cldice_loss(blank, blank, torch.ones(1, 1, 32, 32))) == 0.0))
    x = soft.clone().requires_grad_(True)
    (soft_dice_loss(x, target, mask) + soft_cldice_loss(x, target, mask)).backward()
    checks.append(("gradients flow and are finite", bool(torch.isfinite(x.grad).all()) and float(x.grad.abs().sum()) > 0))
    # recognition loss (M2): a random (untrained) recogniser is enough to check the plumbing
    charset = line_recognizer.Charset("abc ")
    recognizer = line_recognizer.LineCRNN(len(charset)).eval()
    for p in recognizer.parameters():
        p.requires_grad_(False)
    sr = StrokeRestorer()
    line = (torch.rand(95, 300) * 0.3 + 0.7, charset.encode("ab c"), "ab c", "test")
    rec = recognition_loss(sr, recognizer, [line, line], torch.device("cpu"))
    rec.backward()
    grad = sr.body[-1].weight.grad
    checks.append(("recognition loss finite, reaches the SR model, recogniser stays frozen",
                   bool(torch.isfinite(rec)) and grad is not None and float(grad.abs().sum()) > 0
                   and all(p.grad is None for p in recognizer.parameters())))
    before, after = recognition_cer(sr, recognizer, charset, [line], torch.device("cpu"))
    checks.append(("recogniser CER of input and output computed", 0.0 <= before <= 1.0 and 0.0 <= after <= 1.0))
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"values: Dice blurred {float(soft_dice_loss(soft, target, mask)):.3f} | clDice blurred "
          f"{float(soft_cldice_loss(soft, target, mask)):.3f}, broken {float(soft_cldice_loss(broken, target, mask)):.3f}, "
          f"merged {float(soft_cldice_loss(merged, target, mask)):.3f}")
    if not all(ok for _, ok in checks):
        raise SystemExit("self-test FAILED")
    print("self-test passed")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
