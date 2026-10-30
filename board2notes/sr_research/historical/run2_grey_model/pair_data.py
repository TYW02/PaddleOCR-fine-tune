"""Shared, torch-free helpers for the paired super-resolution (stroke restoration) experiment.

Kept free of torch so every data decision (normalisation, patch sampling, splits, metrics) can be
tested on its own. Used by train_accepted_pairs.py, make_airdrop_test_pairs.py and the notebook.

Images are RGB uint8 arrays on disk; inside the model they are float32 in [0, 1].
"""
from __future__ import annotations

import hashlib
import random

import cv2
import numpy as np

# ---------------------------------------------------------------------------------------------
# 1. Photometric normalisation ("flatten the board")
# ---------------------------------------------------------------------------------------------
# The close and the distant photo of a pair were taken with different exposure and white balance.
# Pixel losses then mostly measure that tone difference instead of the strokes. Each image is
# divided by an estimate of its own board background, so the board becomes ~1.0 (white) and ink is
# measured relative to it. Input and target are normalised INDEPENDENTLY (the target is never used
# to normalise the input), so exactly the same step can be applied to a new photo at inference.

BG_WORK_SIDE = 256      # background is estimated on a copy this size (long side)
BG_CLOSE = 9            # closing kernel at that size: removes dark strokes, keeps the board
BG_SIGMA = 6.0          # smooth the background estimate (gradual lighting only)


def board_background(rgb: np.ndarray) -> np.ndarray:
    """Estimate of the board's own colour at every pixel (float32 RGB, >= 1): strokes removed, smoothed."""
    img = rgb.astype(np.float32)
    h, w = img.shape[:2]
    s = min(1.0, BG_WORK_SIDE / max(h, w))
    small = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    k = max(3, int(round(BG_CLOSE * min(1.0, max(small.shape[:2]) / BG_WORK_SIDE))) | 1)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))   # dark ink removed
    bg = cv2.GaussianBlur(bg, (0, 0), BG_SIGMA * min(1.0, max(small.shape[:2]) / BG_WORK_SIDE) + 0.5)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.maximum(bg, 1.0).astype(np.float32)


def normalise_board(rgb: np.ndarray) -> np.ndarray:
    """uint8 RGB -> float32 RGB in [0, 1] with the board background flattened to ~1.0."""
    return np.clip(rgb.astype(np.float32) / board_background(rgb), 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------------------------
# 1b. Greyscale
# ---------------------------------------------------------------------------------------------
# Run 1 (colour) learned mostly to recolour the 17 Oct blue marker: distant photos show it dark and
# dull, close-ups bright blue. That raised PSNR on the 17 Oct board and failed on black-marker boards.
# OCR does not need colour, so the model now sees grey only. The grey value is repeated in all three
# channels so the network architecture and the tiling code stay unchanged.
LUMA = np.array([0.299, 0.587, 0.114], np.float32)


def to_grey3(rgb01: np.ndarray) -> np.ndarray:
    """float RGB in [0, 1] -> the same grey value in all three channels."""
    g = rgb01.astype(np.float32) @ LUMA
    return np.repeat(g[..., None], 3, axis=2)


def denormalise_grey(norm_grey3: np.ndarray, background_rgb: np.ndarray) -> np.ndarray:
    """Undo normalise_board for a grey result: multiply by the grey board background, so the image
    gets back its original brightness (PaddleOCR reads images in their natural tones). -> uint8 RGB."""
    g = norm_grey3[..., 0] * (background_rgb.astype(np.float32) @ LUMA)
    g = np.rint(np.clip(g, 0, 255)).astype(np.uint8)
    return np.repeat(g[..., None], 3, axis=2)


# ---------------------------------------------------------------------------------------------
# 2. Masks: where the canvas holds real photo pixels, and where there is ink
# ---------------------------------------------------------------------------------------------
def valid_mask(rgb: np.ndarray, margin: int = 4, min_area: float = 0.002) -> np.ndarray:
    """False where the canvas is warp fill, True where it holds real photo pixels.

    prepare_real_pairs.warp_crop fills everything outside the photo with ONE exact colour (the photo's
    median board colour). Fill is therefore: pixels of exactly that colour, in a region connected to
    the canvas edge and of noticeable size. (A "flat 5x5 window" test is not enough: heavily enlarged
    inputs have naturally flat board areas.)
    """
    h, w = rgb.shape[:2]
    edge = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]]).reshape(-1, 3)
    colours, counts = np.unique(edge, axis=0, return_counts=True)
    fill = colours[counts.argmax()]
    same = np.all(rgb == fill, axis=2).astype(np.uint8)
    # fill is a SOLID block; smooth enlarged board areas only hit the exact colour in specks and thin
    # chains, which an opening removes
    same = cv2.morphologyEx(same, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(same, connectivity=4)
    bad = np.zeros((h, w), np.uint8)
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        touches_edge = x == 0 or y == 0 or x + cw == w or y + ch == h
        if touches_edge and area >= min_area * h * w:
            bad[lab == i] = 1
    if bad.any():
        bad = cv2.dilate(bad, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))   # cubic-warp ringing
    return bad == 0


def ink_mask(norm_rgb: np.ndarray, thresh: float = 0.75, grow: int = 3) -> np.ndarray:
    """True on (and just around) strokes of a normalised image."""
    ink = (norm_rgb.mean(axis=2) < thresh).astype(np.uint8)
    if grow:
        ink = cv2.dilate(ink, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))
    return ink.astype(bool)


# ---------------------------------------------------------------------------------------------
# 3. Patch sampling
# ---------------------------------------------------------------------------------------------
def sample_patch(rng: random.Random, inp: np.ndarray, tgt: np.ndarray, size: int,
                 ok: np.ndarray, ink: np.ndarray, ink_prob: float = 0.75, tries: int = 20):
    """One size x size patch (y, x). With probability ink_prob it is centred on a stroke pixel,
    otherwise anywhere. Patches that touch warp fill (ok == False) are rejected and redrawn."""
    h, w = tgt.shape[:2]
    size = min(size, h, w)
    ys, xs = np.nonzero(ink)
    for _ in range(tries):
        if len(ys) and rng.random() < ink_prob:
            k = rng.randrange(len(ys))
            y = int(np.clip(ys[k] - size // 2, 0, h - size))
            x = int(np.clip(xs[k] - size // 2, 0, w - size))
        else:
            y, x = rng.randint(0, h - size), rng.randint(0, w - size)
        if ok[y:y + size, x:x + size].mean() > 0.98:
            return y, x, size
    return y, x, size          # fall back to the last draw rather than loop forever


def fixed_patches(key: str, tgt_norm: np.ndarray, ok: np.ndarray, size: int, n: int = 8):
    """Deterministic ink-centred patches for validation: same patches every epoch and every run."""
    rng = random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:12], 16))
    ink = ink_mask(tgt_norm, grow=0) & ok
    return [sample_patch(rng, tgt_norm, tgt_norm, size, ok, ink, ink_prob=1.0) for _ in range(n)]


# ---------------------------------------------------------------------------------------------
# 4. Splits: hold out WHOLE input photos
# ---------------------------------------------------------------------------------------------
# Splitting by line let every input photo appear in train, val and test, so the test measured "new
# lines from photos the model has already seen" (same blur, noise, exposure). Here whole photos are
# held out, one far / side / medium view per split. Synthetic pairs have no input photo (they are
# degraded copies of a clear target) and are used for training only.
TEST_INPUT_PHOTOS = {"IMG_4748.HEIC", "IMG_4750.HEIC", "IMG_4697.HEIC"}   # far, side, medium
VAL_INPUT_PHOTOS = {"IMG_4736.HEIC", "IMG_4705.HEIC", "IMG_4673.HEIC"}    # far, side, medium


def assign_photo_splits(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        if row["pair_type"] != "real":
            split = "train"
        elif row["input_photo"] in TEST_INPUT_PHOTOS:
            split = "test"
        elif row["input_photo"] in VAL_INPUT_PHOTOS:
            split = "val"
        else:
            split = "train"
        out.append({**row, "split": split})
    return out


def spp_band(value) -> str:
    """Source pixels per target pixel (linear): how much the input had to be enlarged."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if v < 0.15:
        return "<0.15 (>6.7x)"
    if v < 0.30:
        return "0.15-0.30 (3.3-6.7x)"
    return ">=0.30 (<3.3x)"


# ---------------------------------------------------------------------------------------------
# 5. Metrics
# ---------------------------------------------------------------------------------------------
def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    d = (a.astype(np.float64) - b.astype(np.float64)) ** 2
    if mask is not None:
        if not mask.any():
            return float("nan")
        d = d[mask]
    return float(-10.0 * np.log10(max(d.mean(), 1e-10)))


def to_uint8(rgb01: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(rgb01, 0, 1) * 255).astype(np.uint8)


def ink_correlation(input_rgb: np.ndarray, target_rgb: np.ndarray) -> float:
    """Same measure as prepare_real_pairs.ink_correlation (dark-stroke structure after removing gradual
    board lighting), so Airdrop pairs are judged on the same scale as the 17 Oct pairs. RGB uint8."""
    a = cv2.cvtColor(input_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    b = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    kernel = max(3, (min(a.shape) // 4) | 1)
    ink_a = np.maximum(cv2.GaussianBlur(a, (kernel, kernel), 0) - a, 0)
    ink_b = np.maximum(cv2.GaussianBlur(b, (kernel, kernel), 0) - b, 0)
    ink_a = (ink_a - ink_a.mean()).ravel()
    ink_b = (ink_b - ink_b.mean()).ravel()
    return float(np.dot(ink_a, ink_b) / (np.linalg.norm(ink_a) * np.linalg.norm(ink_b) + 1e-9))
