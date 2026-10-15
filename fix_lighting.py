#!/usr/bin/env python3
"""
1. Install dependencies


pip install opencv-python numpy
2. Run it on an image


cd ~/Downloads

python fix_lighting.py in.jpg out.jpg
## python fix_lighting.py example1.png example1_fixed.png ##
This writes the lighting-corrected image to out.jpg.

3. Other useful modes
# Just see the metrics without editing the file:
python fix_lighting.py in.jpg --analyze

# Gentler correction (useful before cropping/perspective-warp steps):
python fix_lighting.py in.jpg out.jpg --strength light

# Visual before/field/after comparison strip:
python fix_lighting.py in.jpg out.jpg --debug debug.jpg
###
fix_lighting.py - Lighting normalisation for document / whiteboard photos.

One preprocessing problem, one module. Handles the four faces of "bad lighting":

    1. globally too dim        -> illumination field scaled to a target paper
                                  level (with denoise first, since lifting a
                                  dark frame lifts its sensor noise too)
    2. globally too bright     -> same field, plus a contrast stretch to undo
                                  the wash toward white
    3. uneven illumination     -> flat-fielding: divide the estimated
                                  illumination field out. This is the big win
                                  for documents - a shadow across the page, a
                                  lamp gradient, and the ceiling-light streaks
                                  on a glossy whiteboard are all the same
                                  thing: a smooth multiplicative field.
    4. specular glare          -> detected, suppressed where recoverable, and
                                  REPORTED where not. Blown-out pixels are
                                  physically unrecoverable from a single
                                  frame; we never inpaint text back in,
                                  because invented strokes are worse than a
                                  missing region once an LLM reads the output.

Public API (the shape the future dispatcher needs):

    analyze(img)                 -> dict of metrics + issues + severity, no edit
    fix(img, strength=...)       -> (fixed_img, report_dict)

`strength`:
    'light'   gentle, partial normalisation, meant to run BEFORE geometry
              (corner detection / perspective warp) so page-vs-background
              edges are not flattened away
    'strong'  full normalisation, meant to run AFTER the warp when the frame
              contains only the document
    'auto'    'strong', with the correction amount scaled by measured severity

Classical OpenCV only. No model downloads, no GPU, deterministic.
Input is never binarised; colour in -> colour out, grey in -> grey out.

CLI:
    python fix_lighting.py in.jpg out.jpg
    python fix_lighting.py in.jpg out.jpg --strength light
    python fix_lighting.py in.jpg --analyze
    python fix_lighting.py in.jpg out.jpg --debug debug.jpg
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Tuple

import cv2
import numpy as np

__all__ = ["analyze", "fix", "LightingReport", "debug_strip"]

# ----------------------------------------------------------------------------
# Tunables. Grouped here so the graded harness can sweep them.
# ----------------------------------------------------------------------------

ANALYSIS_MAX_DIM = 1024      # illumination is estimated on a downscaled copy
BG_KERNEL_FRAC = 0.055       # background kernel as a fraction of min(h, w)
BG_KERNEL_MIN = 15

TARGET_PAPER = 236.0         # where the background should land after flat-field
MAX_GAIN = 14.0              # cap on per-pixel lift (noise guard)
MIN_GAIN = 0.2

DIM_BG = 105.0               # paper level below this -> "dim"
BRIGHT_BG = 251.0            # paper level above this -> "bright" (clipping)
UNEVEN_RATIO = 0.13          # illumination spread / level above this -> "uneven"
GLARE_FRAC = 0.002           # >0.2% specular pixels -> "glare"
LOW_INK_CONTRAST = 45.0      # mean ink depth below this -> "low_contrast"

NOISE_SIGMA_DENOISE = 2.6    # estimated sigma above this -> denoise before lift
NLM_MAX_PIXELS = 4_000_000   # above this, use bilateral instead of NL-means

CLAHE_CLIP_STRONG = 1.7
CLAHE_CLIP_LIGHT = 1.1
CLAHE_TILES = 8              # upper bound; actual grid is sized to the image
CLAHE_SKIP_INK_DEPTH = 85.0  # strokes already this deep -> CLAHE adds only
                             # halos and tile texture, so skip it

SPEC_ABOVE_BG = 14           # how far above the local paper level counts as a
                             # highlight rather than just bright paper
SPEC_MIN_V = 232             # ...and it has to be bright in absolute terms
SPEC_MAX_S = 48              # ...and desaturated (a reflection, not ink)
FIELD_GLARE_PAGE_MAX = 248   # a page whose own paper sits above this is simply
                             # over-exposed; no region can be "elevated"
FIELD_GLARE_MARGIN = 6       # how far the field must rise above the paper level
CLIPPED_L = 250              # luminance considered clipped


class LightingReport(dict):
    """A plain dict; subclassed only so it prints readably."""

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return json.dumps(self, indent=2, default=_jsonable)


def _jsonable(o: Any):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------

def _is_gray(img: np.ndarray) -> bool:
    return img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1)


def _luma(img: np.ndarray) -> np.ndarray:
    """Luminance as float32 0..255."""
    if _is_gray(img):
        return img.reshape(img.shape[0], img.shape[1]).astype(np.float32)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _downscale(img: np.ndarray, max_dim: int = ANALYSIS_MAX_DIM) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return img
    s = max_dim / float(m)
    return cv2.resize(img, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                      interpolation=cv2.INTER_AREA)


def _odd(n: int) -> int:
    n = int(n)
    return n if n % 2 == 1 else n + 1


def _noise_sigma(gray: np.ndarray) -> float:
    """
    Immerkaer's fast noise estimate, median variant.

    The textbook version averages |conv|, which text edges inflate badly - a
    sharp receipt reads as a noisy one. Taking the median instead makes the
    estimate robust to the small fraction of pixels that are strokes.
    """
    h, w = gray.shape[:2]
    if h < 5 or w < 5:
        return 0.0
    m = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    conv = np.abs(cv2.filter2D(gray.astype(np.float32), -1, m))[2:-2, 2:-2]
    # For pure Gaussian noise, median|conv| = 4.047 * sigma.
    return float(np.median(conv) / 4.047)


def _polarity(gray_small: np.ndarray) -> str:
    """
    Guess 'dark_ink' (dark text on light paper) or 'light_ink' (chalkboard).

    Histogram-tail tests fail badly here: faint handwriting contributes a tiny
    dark tail while a single glare streak contributes a huge bright one, so a
    glossy whiteboard reads as white-on-black and the whole correction inverts.

    Instead compare thin-structure energy. Black-hat (closing - image) responds
    to thin DARK features, top-hat (image - opening) to thin BRIGHT ones. Large
    smooth glare blobs are wider than the kernel, so they are absorbed into the
    morphological background and do not pollute either residual.
    """
    h, w = gray_small.shape[:2]
    k = _odd(max(7, int(min(h, w) * 0.02)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    g = cv2.GaussianBlur(gray_small.astype(np.float32), (0, 0), 1.0)
    blackhat = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, ker)
    tophat = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, ker)
    # Compare the strong responses only; low-level texture is present in both.
    dark = float(blackhat[blackhat > np.percentile(blackhat, 99)].sum())
    light = float(tophat[tophat > np.percentile(tophat, 99)].sum())
    return "light_ink" if light > dark * 1.6 else "dark_ink"


def _background_field(gray_small: np.ndarray, polarity: str) -> np.ndarray:
    """
    Estimate the smooth illumination field: the image with the ink removed.

    Morphological closing with a kernel wider than the thickest stroke erases
    dark text and leaves the paper; a blur then smooths the residue. For
    light-on-dark we open instead of close.
    """
    h, w = gray_small.shape[:2]
    k = _odd(max(BG_KERNEL_MIN, int(min(h, w) * BG_KERNEL_FRAC)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    op = cv2.MORPH_CLOSE if polarity == "dark_ink" else cv2.MORPH_OPEN
    # Pre-smooth first: closing dilates, so on grainy paper it latches onto the
    # brightest speck under the kernel and biases the whole field upward by
    # several levels - enough to report a clean page as over-exposed.
    src = cv2.GaussianBlur(gray_small.astype(np.float32), (0, 0), 1.2)
    bg = cv2.morphologyEx(src, op, ker)
    bg = cv2.GaussianBlur(bg, (_odd(k), _odd(k)), 0)
    return np.maximum(bg, 1.0)


def _specular_masks(img: np.ndarray,
                    gray: np.ndarray,
                    bg: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (specular, clipped, unrecoverable) as bool arrays.

    A highlight is defined RELATIVE to the paper level, never absolutely:
    white paper is bright and desaturated too, so an absolute threshold flags
    every clean receipt as wall-to-wall glare.

    Two detectors, because they fail in opposite situations:

      local  - pixel brighter than its own local background. Catches small,
               sharp highlights.
      field  - the background FIELD itself sitting far above the page's paper
               level. Catches large, smooth blowouts, which the local test
               misses entirely: a big soft blob gets absorbed into the
               background estimate, so nothing ever looks brighter than it.

    unrecoverable = specular AND clipped: the surface mirrored the lamp into
    the sensor and whatever stroke was there is simply not in the file.
    """
    clipped = gray >= CLIPPED_L
    paper_ref = float(np.median(bg))

    local_hot = (gray >= bg + SPEC_ABOVE_BG) & (gray >= SPEC_MIN_V)

    # The field test only has meaning while the page as a whole is NOT near
    # clipping. If the whole sheet is pinned at 254 there is no "elevated"
    # region to find - that is global over-exposure, which `bright` covers.
    if paper_ref < FIELD_GLARE_PAGE_MAX:
        field_hot = (bg >= max(paper_ref + FIELD_GLARE_MARGIN,
                               FIELD_GLARE_PAGE_MAX)) & (gray >= SPEC_MIN_V)
    else:
        field_hot = np.zeros_like(local_hot)

    hot = local_hot | field_hot

    if _is_gray(img):
        spec = hot
    else:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        spec = hot & (hsv[:, :, 1] <= SPEC_MAX_S)
    return spec, clipped, (spec & clipped)


def _blown_over_text(gray: np.ndarray, bg: np.ndarray, lost: np.ndarray) -> float:
    """
    Percentage of the blown-out area that falls inside the region where writing
    was found. Glare on an empty corner is harmless; the same glare across the
    text is the difference between a usable photo and a retake.

    Text under a blowout has already been erased, so it cannot be detected
    directly - we use the bounding box of the ink that IS still visible as a
    stand-in for "where the writing lives".
    """
    if lost.sum() == 0:
        return 0.0
    depth = bg - gray
    thr = max(6.0, float(np.percentile(depth, 99.5)) * 0.30)
    ink = (depth > thr).astype(np.uint8)
    if ink.sum() < 20:
        return 0.0
    ys, xs = np.nonzero(ink)
    y0, y1 = int(np.percentile(ys, 1)), int(np.percentile(ys, 99))
    x0, x1 = int(np.percentile(xs, 1)), int(np.percentile(xs, 99))
    box = np.zeros_like(ink, dtype=bool)
    box[y0:y1 + 1, x0:x1 + 1] = True
    return float((lost & box).sum() / max(1, lost.sum()) * 100.0)


def _ink_stats(gray: np.ndarray, bg: np.ndarray) -> Tuple[float, float]:
    """
    Mean ink depth (how far strokes sit below the local background) and the
    fraction of pixels that look like ink. Measured against the illumination
    field, so a dim photo of good handwriting doesn't read as 'no text'.
    """
    depth = bg - gray
    thr = max(6.0, float(np.percentile(depth, 99.5)) * 0.30)
    ink = depth > thr
    frac = float(ink.mean())
    if frac < 1e-5:
        return 0.0, frac
    return float(depth[ink].mean()), frac


# ----------------------------------------------------------------------------
# analyze()
# ----------------------------------------------------------------------------

def analyze(img: np.ndarray, polarity: str = "dark_ink") -> LightingReport:
    """
    Measure the lighting of `img` without modifying it.

    polarity : 'dark_ink' (default) | 'light_ink' | 'auto'
        Nearly every real input - receipts, paper, signs, whiteboards - is dark
        marks on a light surface, so that is the default. 'auto' guesses, and
        is only worth using if chalkboards are genuinely in scope: guessing
        wrong inverts the entire correction.

    Returns a dict with raw metrics, a list of `issues`, a 0..1 `severity`,
    and a `recommend` block the dispatcher can act on.
    """
    if img is None or img.size == 0:
        raise ValueError("analyze() got an empty image")
    if polarity not in ("dark_ink", "light_ink", "auto"):
        raise ValueError(f"unknown polarity {polarity!r}")

    small = _downscale(img)
    gray_s = _luma(small)

    p1, p5, p50, p95, p99 = np.percentile(gray_s, [1, 5, 50, 95, 99])
    mean = float(gray_s.mean())

    pol = _polarity(gray_s) if polarity == "auto" else polarity
    bg = _background_field(gray_s, pol)

    bg_p5, bg_p50, bg_p95 = np.percentile(bg, [5, 50, 95])
    illum_spread = float(bg_p95 - bg_p5)
    illum_ratio = illum_spread / max(1.0, float(bg_p50))

    spec_s, clip_s, lost_s = _specular_masks(small, gray_s, bg)
    spec_frac = float(spec_s.mean())
    clip_frac = float(clip_s.mean())
    lost_frac = float(lost_s.mean())
    dark_frac = float((gray_s <= 5).mean())

    ink_depth, ink_frac = _ink_stats(gray_s, bg)
    sigma = _noise_sigma(gray_s)

    # Exposure is judged on the PAPER level, not the overall median. A clean
    # receipt legitimately has a median near 248; that is not over-exposure.
    issues = []
    if bg_p50 < DIM_BG:
        issues.append("dim")
    if bg_p50 > BRIGHT_BG:
        issues.append("bright")
    if illum_ratio > UNEVEN_RATIO:
        issues.append("uneven")
    if spec_frac > GLARE_FRAC:
        issues.append("glare")
    if ink_frac > 1e-4 and ink_depth < LOW_INK_CONTRAST:
        issues.append("low_contrast")
    if sigma > NOISE_SIGMA_DENOISE:
        issues.append("noisy")

    parts = [
        min(1.0, max(0.0, (DIM_BG - bg_p50) / DIM_BG)),
        min(1.0, max(0.0, (bg_p50 - BRIGHT_BG) / (255.0 - BRIGHT_BG))),
        min(1.0, illum_ratio / 0.45),
        min(1.0, spec_frac / 0.08),
        min(1.0, max(0.0, (LOW_INK_CONTRAST - ink_depth) / LOW_INK_CONTRAST)),
    ]
    parts = [float(p) for p in parts]
    severity = float(min(1.0, max(parts) * 0.75 + (sum(parts) - max(parts)) * 0.12))

    return LightingReport({
        "issues": issues,
        "severity": round(severity, 3),
        "polarity": pol,
        "metrics": {
            "mean": round(mean, 2),
            "p1": round(float(p1), 2),
            "p5": round(float(p5), 2),
            "p50": round(float(p50), 2),
            "p95": round(float(p95), 2),
            "p99": round(float(p99), 2),
            "paper_level": round(float(bg_p50), 2),
            "paper_low": round(float(bg_p5), 2),
            "paper_high": round(float(bg_p95), 2),
            "illum_spread": round(illum_spread, 2),
            "illum_ratio": round(illum_ratio, 4),
            "specular_frac": round(spec_frac, 5),
            "clipped_frac": round(clip_frac, 5),
            "unrecoverable_frac": round(lost_frac, 5),
            "black_frac": round(dark_frac, 5),
            "ink_depth": round(ink_depth, 2),
            "ink_frac": round(ink_frac, 5),
            "noise_sigma": round(sigma, 2),
        },
        "recommend": {
            "needs_fix": bool(issues),
            "strength": "strong" if severity > 0.25 else "light",
            "denoise": bool(sigma > NOISE_SIGMA_DENOISE),
            "unrecoverable_pct": round(lost_frac * 100.0, 2),
            "retake_advised": bool(lost_frac > 0.02),
        },
        "shape": {"h": int(img.shape[0]), "w": int(img.shape[1]),
                  "gray": bool(_is_gray(img))},
    })


# ----------------------------------------------------------------------------
# fix()
# ----------------------------------------------------------------------------

def _denoise(img: np.ndarray, sigma: float) -> np.ndarray:
    """
    Edge-preserving denoise sized to the measured noise. Runs BEFORE any lift,
    because brightening a dark frame brightens its grain by the same factor and
    OCR reads that grain as speckle.
    """
    h = float(np.clip(sigma * 1.6, 3, 18))
    if img.shape[0] * img.shape[1] <= NLM_MAX_PIXELS:
        if _is_gray(img):
            return cv2.fastNlMeansDenoising(
                img.reshape(img.shape[0], img.shape[1]), None, h, 7, 21)
        return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)
    # Full-res phone frames: NL-means is too slow, bilateral is close enough.
    return cv2.bilateralFilter(img, 7, float(np.clip(sigma * 4, 15, 90)), 9)


def _white_balance(img: np.ndarray, ref_mask: np.ndarray) -> np.ndarray:
    """
    Neutralise a colour cast using the paper itself as the white reference.
    Documents hand us something most scenes don't: a large region we know is
    meant to be neutral.
    """
    if _is_gray(img) or ref_mask.sum() < 50:
        return img
    f = img.astype(np.float32)
    means = [float(f[:, :, c][ref_mask].mean()) for c in range(3)]
    g = float(np.mean(means))
    out = f.copy()
    for c in range(3):
        if means[c] > 1.0:
            out[:, :, c] = f[:, :, c] * float(np.clip(g / means[c], 0.7, 1.4))
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_gain(img: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """
    Apply a per-pixel luminance gain, keeping colour ratios. Scaling RGB beats
    a LAB round-trip here: same luminance result, no desaturation.
    """
    if _is_gray(img):
        base = img.reshape(img.shape[0], img.shape[1]).astype(np.float32)
        return np.clip(base * gain, 0, 255).astype(np.uint8)
    return np.clip(img.astype(np.float32) * gain[:, :, None], 0, 255).astype(np.uint8)


def fix(img: np.ndarray,
        strength: str = "auto",
        white_balance: bool = True,
        denoise: str | bool = "auto",
        polarity: str = "dark_ink",
        report: Dict[str, Any] | None = None) -> Tuple[np.ndarray, LightingReport]:
    """
    Normalise the lighting of `img`.

    strength      : 'auto' | 'light' | 'strong'  (see module docstring)
    white_balance : neutralise colour cast using the paper as white reference
    denoise       : 'auto' | True | False
    polarity      : 'dark_ink' (default) | 'light_ink' | 'auto'

    Returns (image, report). Same dtype and channel count as the input, never
    binarised.
    """
    if img is None or img.size == 0:
        raise ValueError("fix() got an empty image")
    if strength not in ("auto", "light", "strong"):
        raise ValueError(f"unknown strength {strength!r}")

    rep = LightingReport(dict(report) if report else dict(analyze(img, polarity)))
    sev = float(rep["severity"])
    mode = "strong" if strength == "auto" else strength
    pol = rep["polarity"]
    actions = []
    work = img.copy()

    # --- 1. denoise before any lift -----------------------------------------
    want_dn = (denoise is True) or (denoise == "auto" and rep["recommend"]["denoise"])
    if want_dn:
        sigma = float(rep["metrics"]["noise_sigma"])
        work = _denoise(work, sigma)
        actions.append(f"denoise(sigma={sigma:.1f})")

    gray_full = _luma(work)
    small = _downscale(work)
    gray_s = _luma(small)

    # --- 2. illumination field: estimated small, applied full-res ------------
    bg_s = _background_field(gray_s, pol)
    spec_s, clip_s, lost_s = _specular_masks(small, gray_s, bg_s)

    # A blown highlight must not drag the field to 255 and crush the paper
    # around it, so cap the field at a high percentile of itself.
    cap = float(np.percentile(bg_s, 99.0))
    bg_s = np.minimum(bg_s, max(cap, 1.0))

    bg = cv2.resize(np.maximum(bg_s, 1.0), (work.shape[1], work.shape[0]),
                    interpolation=cv2.INTER_LINEAR)
    bg = np.maximum(bg, 1.0)

    spec_full = cv2.resize(spec_s.astype(np.uint8),
                           (work.shape[1], work.shape[0]),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
    clip_full = gray_full >= CLIPPED_L
    lost_full = spec_full & clip_full
    lost_pct = float(lost_full.mean() * 100.0)

    # How much of the blown area sits where the writing is? Glare on an empty
    # corner of a whiteboard costs nothing; the same glare across the text is
    # the difference between a usable photo and a retake.
    risk_pct = _blown_over_text(gray_s, bg_s, (spec_s & clip_s))

    # Flat-field: divide the illumination out.
    if pol == "dark_ink":
        gain_flat = TARGET_PAPER / bg
    else:
        gain_flat = (255.0 - TARGET_PAPER) / np.maximum(255.0 - bg, 1.0)

    # 'light' mode corrects only partially, so the page-vs-background gradient
    # that corner detection relies on is not flattened into nothing.
    alpha = 1.0 if mode == "strong" else float(np.clip(0.35 + 0.35 * sev, 0.3, 0.7))
    gain = np.clip(1.0 + (gain_flat - 1.0) * alpha, MIN_GAIN, MAX_GAIN).astype(np.float32)

    work = _apply_gain(work, gain)
    actions.append(f"flat_field(alpha={alpha:.2f},target={TARGET_PAPER:.0f},"
                   f"max_gain={float(gain.max()):.1f})")

    # --- 3. white balance off the paper --------------------------------------
    if white_balance and not _is_gray(work):
        g2 = _luma(work)
        ref = (g2 > np.percentile(g2, 70)) & (~spec_full)  # highlights are not
        work = _white_balance(work, ref)                   # a colour reference
        actions.append("white_balance(paper_ref)")

    # --- 4. exposure trim ----------------------------------------------------
    # After a full flat-field this is nearly a no-op; it earns its keep in
    # 'light' mode, where only part of the field was divided out.
    g3 = _luma(work)
    bg3 = _background_field(_downscale(g3.astype(np.uint8)), pol)
    paper = float(np.median(bg3))
    if paper > 2.0 and abs(paper - TARGET_PAPER) > 6.0:
        g = float(np.clip(np.log(TARGET_PAPER / 255.0) /
                          np.log(np.clip(paper, 2.0, 254.0) / 255.0), 0.4, 2.5))
        lut = np.clip(((np.arange(256) / 255.0) ** g) * 255.0, 0, 255).astype(np.uint8)
        work = cv2.LUT(work, lut)
        actions.append(f"gamma({g:.2f})")

    # --- 5. contrast stretch, anchored to the paper level --------------------
    # Anchored, not stretched to full range. Mapping p99.6 -> 255 looks fine
    # once, but it is not idempotent: each pass shifts the paper higher, so a
    # frame that goes through lighting twice (which the combined pipeline will
    # do - light before geometry, strong after) marches toward a blown page.
    # Pinning ink -> 0 and paper -> TARGET_PAPER gives the same contrast and
    # lands on the same place no matter how many times it runs.
    g4 = _luma(work)
    valid = ~clip_full
    if valid.sum() < g4.size * 0.05:
        valid = np.ones_like(clip_full)
    sample = g4[valid]
    if sample.size > 100:
        lo = float(np.percentile(sample, 0.4))
        paper_now = float(np.median(_background_field(_luma(_downscale(work)), pol)))
        if paper_now - lo > 12.0 and (lo > 3.0 or abs(paper_now - TARGET_PAPER) > 4.0):
            scale = TARGET_PAPER / (paper_now - lo)
            lut = np.clip((np.arange(256, dtype=np.float32) - lo) * scale,
                          0, 255).astype(np.uint8)
            work = cv2.LUT(work, lut)
            actions.append(f"stretch(ink={lo:.0f}->0,paper={paper_now:.0f}->{TARGET_PAPER:.0f})")

    # --- 6. local contrast, only if the page still needs it ------------------
    # Flat-fielding already normalised local brightness, so CLAHE is a top-up,
    # not a staple. Run unconditionally and it rings a bright halo around every
    # stroke and stamps its tile grid across empty paper - both of which OCR
    # sees as structure that is not there.
    small_now = _luma(_downscale(work))
    bg_now = _background_field(small_now, pol)
    depth_now, _ = _ink_stats(small_now, bg_now)
    sigma_now = _noise_sigma(small_now)

    if depth_now < CLAHE_SKIP_INK_DEPTH:
        base = CLAHE_CLIP_STRONG if mode == "strong" else CLAHE_CLIP_LIGHT
        clip_limit = float(np.clip(base * (1.0 - sigma_now / 9.0), 0.6, base))
        # Tiles must be comfortably larger than a glyph, or CLAHE treats the
        # text itself as the local histogram and halos it.
        tile = int(np.clip(min(work.shape[:2]) // 160, 2, CLAHE_TILES))
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
        if _is_gray(work):
            work = clahe.apply(work.reshape(work.shape[0], work.shape[1]))
        else:
            lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            work = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        actions.append(f"clahe(clip={clip_limit:.2f},tiles={tile})")
    else:
        actions.append(f"clahe(skipped,ink_depth={depth_now:.0f})")

    # --- 7. report -----------------------------------------------------------
    after = analyze(work, pol)
    return work, LightingReport({
        "mode": mode,
        "actions": actions,
        "before": rep,
        "after": after,
        "glare": {
            "unrecoverable_pct": round(lost_pct, 2),
            "blown_over_text_pct": round(risk_pct, 1),
            "note": ("blown-out region(s) present; text there cannot be recovered "
                     "from a single frame - retake advised"
                     if lost_pct > 2.0 else
                     "no significant unrecoverable region"),
            "retake_advised": bool(lost_pct > 2.0),
        },
        "improved": {
            "paper_level": [rep["metrics"]["paper_level"], after["metrics"]["paper_level"]],
            "ink_depth": [rep["metrics"]["ink_depth"], after["metrics"]["ink_depth"]],
            "illum_ratio": [rep["metrics"]["illum_ratio"], after["metrics"]["illum_ratio"]],
            "severity": [rep["severity"], after["severity"]],
        },
    })


# ----------------------------------------------------------------------------
# Debug strip
# ----------------------------------------------------------------------------

def debug_strip(before: np.ndarray, after: np.ndarray, rep: Dict[str, Any]) -> np.ndarray:
    """Before / illumination field / after, side by side."""
    def to_bgr(x):
        return x if not _is_gray(x) else cv2.cvtColor(
            x.reshape(x.shape[0], x.shape[1]), cv2.COLOR_GRAY2BGR)

    b, a = to_bgr(before), to_bgr(after)
    pol = rep.get("before", {}).get("polarity", "dark_ink")
    field = _background_field(_luma(_downscale(before)), pol)
    field = cv2.normalize(field, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    field = cv2.applyColorMap(cv2.resize(field, (before.shape[1], before.shape[0])),
                              cv2.COLORMAP_INFERNO)

    H = 520
    def rs(x):
        s = H / x.shape[0]
        return cv2.resize(x, (max(1, int(x.shape[1] * s)), H))

    panes = [rs(b), rs(field), rs(a)]
    strip = np.hstack(panes)
    x = 0
    for p, lab in zip(panes, ["before", "illumination field", "after"]):
        for col, th in (((0, 0, 0), 4), ((255, 255, 255), 1)):
            cv2.putText(strip, lab, (x + 10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, col, th, cv2.LINE_AA)
        x += p.shape[1]
    return strip


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def _read(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"could not read image: {path}")
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fix lighting in a document photo.")
    ap.add_argument("input")
    ap.add_argument("output", nargs="?")
    ap.add_argument("--strength", default="auto", choices=["auto", "light", "strong"])
    ap.add_argument("--analyze", action="store_true", help="print metrics only")
    ap.add_argument("--no-wb", action="store_true", help="skip white balance")
    ap.add_argument("--denoise", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--polarity", default="dark_ink",
                    choices=["dark_ink", "light_ink", "auto"],
                    help="dark marks on light surface (default), the inverse "
                         "(chalkboard), or guess")
    ap.add_argument("--debug", metavar="PATH", help="write a before/field/after strip")
    a = ap.parse_args(argv)

    img = _read(a.input)

    if a.analyze:
        print(json.dumps(analyze(img, a.polarity), indent=2, default=_jsonable))
        return 0
    if not a.output:
        ap.error("output path required unless --analyze")

    dn = {"auto": "auto", "on": True, "off": False}[a.denoise]
    out, rep = fix(img, strength=a.strength, white_balance=not a.no_wb,
                   denoise=dn, polarity=a.polarity)

    d = os.path.dirname(os.path.abspath(a.output))
    os.makedirs(d, exist_ok=True)
    cv2.imwrite(a.output, out)
    if a.debug:
        cv2.imwrite(a.debug, debug_strip(img, out, rep))

    print(json.dumps(rep, indent=2, default=_jsonable))
    if rep["glare"]["retake_advised"]:
        print(f"\n!! {rep['glare']['unrecoverable_pct']}% of the frame is blown out. "
              f"Text there is gone; a retake is the only fix.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
