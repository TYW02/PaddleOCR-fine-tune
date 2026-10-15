#!/usr/bin/env python3
"""
fix_rotation.py -- preprocessing module #2: in-plane rotation / skew correction.

One job only: take a photo whose text is tilted (a few degrees of skew, a
diagonal scrawl, or a full sideways 90 deg) and return the same photo with the
text sitting level -- parallel to the bottom of the frame.

Deliberately NOT in scope:
  * perspective / camera angle  -> that is the other module
  * 180 deg upside-down         -> geometry cannot tell; leave it to the OCR stage
  * blur, lighting, denoise     -> their own modules

Deliberately classical (OpenCV + NumPy only). No OCR engine, no neural net.
This stage should be cheap and dumb: find the dominant writing direction and
rotate. Recognising the letters is the OCR stage's job.

Public API
----------
    img_out, info = fix_rotation(img_bgr)

    info = {
        "angle":       float,   # degrees applied, +ve = counter-clockwise
        "confidence":  float,   # 0..1, how sure we are text is horizontal now
        "method":      str,     # which estimator won
        "applied":     bool,    # False if angle was below the dead-zone
        "ink_ratio":   float,   # fraction of pixels judged to be ink
        "candidates":  dict,    # every estimator's raw answer, for debugging
    }

CLI
---
    python fix_rotation.py photo.jpg -o fixed.jpg --debug panel.png
    python fix_rotation.py folder/ -o outdir/ --debug-dir dbg/
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# tunables
# ----------------------------------------------------------------------------

WORK_MAX_DIM = 1100      # analysis resolution; output is always full-res
BG_KERNEL_FRAC = 0.035   # median kernel for local-background estimate
MIN_COMP_AREA_FRAC = 2.5e-5   # components smaller than this are specks
MIN_COMP_AREA_ABS = 12
STRENGTH_KEEP = 0.30     # keep components at >= this * robust reference strength
COARSE_STEP = 1.0        # deg
FINE_SPAN = 1.5          # deg either side of coarse winner
FINE_STEP = 0.05         # deg
DEAD_ZONE = 0.15         # below this many degrees, don't bother resampling
SNAP_TOL = 0.25          # snap to exact 0/90 within this many degrees


# ----------------------------------------------------------------------------
# stage 1 -- ink mask
# ----------------------------------------------------------------------------

def _downscale(img, max_dim=WORK_MAX_DIM):
    h, w = img.shape[:2]
    s = min(1.0, float(max_dim) / max(h, w))
    if s >= 1.0:
        return img, 1.0
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                      interpolation=cv2.INTER_AREA), s


def _odd(n):
    n = int(n)
    return n + 1 if n % 2 == 0 else n


def _valid_region(bgr):
    """Kept as a hook, but deliberately a no-op.

    An earlier warp (the perspective stage) leaves constant-colour wedges in
    the corners. Those need no special handling: a median filter over a
    constant region returns that same constant, so the response there is
    already zero. Only the wedge *boundary* answers, and it answers as one very
    long, very sparse component, which the elongation filter below removes.

    Detecting the wedges by their flatness was tried and was a mistake -- a
    JPEG-smoothed whiteboard is just as flat, so it erased the writing too.
    """
    return np.ones(bgr.shape[:2], np.uint8)


def ink_mask(bgr, debug=None):
    """Find written strokes on a real-world surface.

    Whiteboards are the hard case: strong illumination gradients, specular
    glare blobs, and half-erased ghost writing. So instead of thresholding
    brightness we compare each pixel against its own *local* background colour
    in Lab space. That kills gradients and glare (both are low-frequency) and
    it picks up coloured marker as readily as black, which a plain grayscale
    threshold does not.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    h, w = lab.shape[:2]

    k = _odd(max(9, BG_KERNEL_FRAC * max(h, w)))
    # median over a window wider than any stroke -> strokes vanish, surface stays
    bg = cv2.medianBlur(lab, min(k, 255))

    d = lab.astype(np.float32) - bg.astype(np.float32)
    # L: only darker-than-background counts as ink (glare is brighter)
    dl = np.clip(-d[:, :, 0], 0, None)
    da = np.abs(d[:, :, 1])
    db = np.abs(d[:, :, 2])
    resp = np.sqrt(dl * dl + da * da + db * db)

    # Divide by the local background lightness. Illumination is multiplicative,
    # so the same pen stroke gives a big absolute difference on the lit side of
    # a board and a small one in shadow. Without this, one global threshold
    # cannot serve both halves and the dim half of the writing is thrown away.
    bgL = np.maximum(bg[:, :, 0].astype(np.float32), 25.0)
    resp *= 128.0 / bgL

    # a light blur closes pen-stroke gaps without merging separate letters
    resp = cv2.GaussianBlur(resp, (3, 3), 0)

    valid = _valid_region(bgr)
    resp *= valid

    # Pixels with an exactly-zero response are not surface at all: they are the
    # constant fill an earlier warp left in the corners, where the median of a
    # constant returns that same constant. Real surface always answers a little,
    # because of sensor noise. Excluding those pixels from the threshold
    # statistics matters more than it sounds -- a big fill region puts a huge
    # spike at zero in the histogram, and the triangle rule then places the
    # threshold down among the noise and calls the whole board ink.
    live = resp > 1e-6
    vals = resp[live]
    if vals.size < 64:
        if debug is not None:
            debug["n_comp"] = 0
        return np.zeros((h, w), np.uint8)

    hi = float(np.percentile(vals, 99.7))
    if hi <= 1e-6:
        if debug is not None:
            debug["n_comp"] = 0
        return np.zeros((h, w), np.uint8)
    r8 = np.clip(resp * (255.0 / hi), 0, 255).astype(np.uint8)

    # Triangle beats Otsu here: ink is a tiny minority of pixels, so the
    # histogram is heavily skewed and Otsu drags the threshold far too low.
    thr, _ = cv2.threshold(r8[live].reshape(-1, 1), 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
    m = ((r8 > thr) & live & (valid > 0)).astype(np.uint8)

    # drop specks, dust, erased ghost text and smudges
    n, lab_im, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    min_area = max(MIN_COMP_AREA_ABS, int(MIN_COMP_AREA_FRAC * h * w))
    maxdim = max(h, w)
    sized = np.zeros(n, bool)
    strengths = np.zeros(n, np.float32)
    areas = np.zeros(n, np.float32)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        # a board frame, a table edge or a ruled line: far too long and far too
        # sparse inside its own bounding box to be a letter
        if max(bw, bh) > 0.55 * maxdim and area / float(bw * bh) < 0.06:
            continue
        sel = lab_im == i
        strengths[i] = float(resp[sel].mean())
        areas[i] = float(area)
        sized[i] = True

    keep = sized.copy()
    if sized.any():
        # A half-erased ghost of the word, or a marker smudge, survives the
        # threshold but is much fainter than live ink. Compare against a robust
        # upper quantile rather than the single strongest component, which one
        # bright glint would otherwise set.
        idx = np.nonzero(sized)[0]
        order = idx[np.argsort(strengths[idx])]
        cw = np.cumsum(areas[order])
        ref = float(strengths[order[int(np.searchsorted(cw, 0.75 * cw[-1]))]])
        weak = sized & (strengths < STRENGTH_KEEP * ref)
        # safety valve: if that would discard most of the writing, the spread is
        # illumination rather than ghosting, so keep everything
        if areas[weak].sum() <= 0.5 * areas[sized].sum():
            keep = sized & ~weak

    out = keep[lab_im].astype(np.uint8)

    if debug is not None:
        debug["resp"] = r8
        debug["thr"] = thr
        debug["raw_mask"] = m
        debug["valid"] = valid
        debug["n_comp"] = int(keep.sum())

    return out


# ----------------------------------------------------------------------------
# stage 2 -- angle estimation
# ----------------------------------------------------------------------------

def _ink_points(mask, max_pts=60000):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    if len(xs) > max_pts:
        idx = np.random.default_rng(0).choice(len(xs), max_pts, replace=False)
        xs, ys = xs[idx], ys[idx]
    p = np.stack([xs, ys], 1).astype(np.float32)
    return p - p.mean(0, keepdims=True)


def _profile_score(pts, angle_deg, bin_px=1.0):
    """How tightly does the ink stack into horizontal bands at this angle?

    Rotating the image CCW by theta sends (x, y) -> y' = -x sin + y cos
    (OpenCV's convention, y pointing down). Project every ink pixel onto y',
    histogram it, and score the concentration. Level text piles all its ink
    into a few rows -- one x-height band per line -- so the score peaks exactly
    when the writing is straight. Sweeping the full 180 deg means a sideways
    photo is found by the same mechanism as a 3 deg skew; there is no separate
    "is it rotated 90" special case to get wrong.
    """
    t = math.radians(angle_deg)
    yp = -pts[:, 0] * math.sin(t) + pts[:, 1] * math.cos(t)

    # Spread each pixel across its two nearest bins by fractional weight.
    # Snapping to whole bins instead looks harmless but is not: at 0, +-45 and
    # other simple angles a grid of integer pixel coordinates projects onto a
    # coarser lattice, so the histogram spikes and the score jumps ~30% for
    # purely arithmetic reasons. That aliasing pulls every answer towards
    # -45/0/+45. Fractional accumulation plus a light smooth removes it.
    b = yp / bin_px
    b -= b.min()
    i0 = np.floor(b).astype(np.int64)
    f = b - i0
    n_bins = int(i0.max()) + 2
    prof = (np.bincount(i0, weights=1.0 - f, minlength=n_bins)
            + np.bincount(i0 + 1, weights=f, minlength=n_bins))
    if len(prof) >= 3:
        prof = np.convolve(prof, np.array([0.25, 0.5, 0.25]), mode="same")
    n = prof.sum()
    if n <= 0:
        return 0.0, 0.0
    conc = float((prof * prof).sum()) / (n * n)          # peakedness
    grad = float((np.diff(prof) ** 2).sum()) / (n * n)   # line separation
    return conc, grad


def _sweep(pts, lo, hi, step, weight_grad=0.5):
    angles = np.arange(lo, hi, step)
    sc = np.empty(len(angles))
    for i, a in enumerate(angles):
        c, g = _profile_score(pts, float(a))
        sc[i] = c + weight_grad * g
    return angles, sc


def _dir_to_correction(vx, vy):
    """Direction vector of the writing -> rotation that levels it.

    Rotating CCW by theta sends y -> -x sin + y cos, so the direction (vx, vy)
    becomes horizontal when theta = atan2(vy, vx). Deriving it this way rather
    than reading an angle straight off cv2.minAreaRect keeps the module
    independent of the OpenCV version, whose rect-angle range changed in 4.5.
    """
    return _wrap90(math.degrees(math.atan2(vy, vx)))


def _minarearect_angle(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) < 5:
        return None
    pts = np.stack([xs, ys], 1).astype(np.float32)
    box = cv2.boxPoints(cv2.minAreaRect(pts))
    # take the longer side of the box as the writing direction: a line, or a
    # block of lines, is wider than it is tall
    e1, e2 = box[1] - box[0], box[2] - box[1]
    e = e1 if np.hypot(*e1) >= np.hypot(*e2) else e2
    return _dir_to_correction(float(e[0]), float(e[1]))


def _pca_angle(pts):
    if pts is None or len(pts) < 5:
        return None
    cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    vx, vy = vecs[:, int(np.argmax(vals))]
    return _dir_to_correction(float(vx), float(vy))


def _centroid_line_angle(mask, min_comp=3):
    """Fit a line through the letter centroids.

    For a single line of handwriting this is the most direct read of the
    baseline, and it is independent of stroke shape -- so it makes a good
    sanity check on the projection sweep rather than a replacement for it.
    """
    n, lab_im, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    if n - 1 < min_comp:
        return None
    pts = cents[1:].astype(np.float32)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
    keep = areas > 0.15 * areas.max()
    pts = pts[keep]
    if len(pts) < min_comp:
        return None
    vx, vy, _, _ = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    return _dir_to_correction(float(vx), float(vy))


def _wrap90(a):
    """Fold an angle into (-90, 90]; orientation is mod 180."""
    a = (a + 90.0) % 180.0 - 90.0
    return 90.0 if a <= -90.0 else a


def _ang_diff(a, b):
    return abs(_wrap90(a - b))


def estimate_angle(mask, debug=None):
    """Return (correction_deg, confidence, method, candidates)."""
    pts = _ink_points(mask)
    if pts is None or len(pts) < 30:
        return 0.0, 0.0, "no-ink", {}

    # coarse sweep over every orientation, then refine the winner
    angles, sc = _sweep(pts, -90.0, 90.0, COARSE_STEP)
    best_i = int(np.argmax(sc))
    coarse = float(angles[best_i])

    fa, fs = _sweep(pts, coarse - FINE_SPAN, coarse + FINE_SPAN + 1e-9, FINE_STEP)
    fine = float(fa[int(np.argmax(fs))])
    fine = _wrap90(fine)

    # confidence: how much better is this than the perpendicular reading?
    perp_i = int(np.argmin(np.abs(((angles - (coarse + 90) + 90) % 180) - 90)))
    s_best, s_perp = float(sc[best_i]), float(sc[perp_i])
    c_perp = 0.0 if s_best <= 0 else max(0.0, 1.0 - s_perp / s_best)

    # ...and how sharp is the peak? a broad peak means a vague answer
    med = float(np.median(sc))
    c_peak = 0.0 if s_best <= med else min(1.0, (s_best - med) / max(s_best, 1e-9) * 2.0)

    cands = {
        "projection": fine,
        "minarearect": _minarearect_angle(mask),
        "pca": _pca_angle(pts),
        "centroid_line": _centroid_line_angle(mask),
    }

    # the sweep is the answer; the others only vote on trust
    agree = [v for k, v in cands.items()
             if k != "projection" and v is not None and _ang_diff(v, fine) < 8.0]
    c_agree = min(1.0, 0.4 + 0.2 * len(agree))

    conf = float(np.clip(0.5 * c_perp + 0.25 * c_peak + 0.25 * c_agree, 0.0, 1.0))

    if debug is not None:
        debug["sweep_angles"] = angles
        debug["sweep_scores"] = sc

    return fine, conf, "projection", cands


# ----------------------------------------------------------------------------
# stage 3 -- rotate
# ----------------------------------------------------------------------------

def _border_colour(bgr, mask_small, scale):
    """Median of the surface, so new corners match the board instead of
    appearing as black wedges that a later binarisation would chew on."""
    h, w = bgr.shape[:2]
    small = cv2.resize(bgr, (mask_small.shape[1], mask_small.shape[0]),
                       interpolation=cv2.INTER_AREA)
    bgpx = small[mask_small == 0]
    if len(bgpx) < 50:
        bgpx = small.reshape(-1, 3)
    return tuple(float(v) for v in np.median(bgpx, 0))


def rotate_keep_all(bgr, angle, border):
    h, w = bgr.shape[:2]
    c = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(c, angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw = int(round(h * sin + w * cos))
    nh = int(round(h * cos + w * sin))
    M[0, 2] += nw / 2.0 - c[0]
    M[1, 2] += nh / 2.0 - c[1]
    return cv2.warpAffine(bgr, M, (nw, nh), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border), M


# ----------------------------------------------------------------------------
# public entry point
# ----------------------------------------------------------------------------

def fix_rotation(bgr, expand=True, crop_to_text=False, margin=0.12,
                 return_debug=False):
    dbg = {}
    small, scale = _downscale(bgr)
    mask = ink_mask(small, dbg)
    ink_ratio = float(mask.mean())

    angle, conf, method, cands = estimate_angle(mask, dbg)

    # snap: a photo that is 0.03 deg off is level, and 89.9 deg is sideways.
    # Snapping avoids a pointless resample that only costs sharpness.
    for target in (-90.0, 0.0, 90.0):
        if abs(angle - target) < SNAP_TOL:
            angle = target
            break

    applied = abs(angle) >= DEAD_ZONE
    border = _border_colour(bgr, mask, scale)

    if applied:
        out, M = rotate_keep_all(bgr, angle, border) if expand else (
            cv2.warpAffine(bgr, cv2.getRotationMatrix2D(
                (bgr.shape[1] / 2.0, bgr.shape[0] / 2.0), angle, 1.0),
                (bgr.shape[1], bgr.shape[0]), flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT, borderValue=border),
            cv2.getRotationMatrix2D((bgr.shape[1] / 2.0, bgr.shape[0] / 2.0),
                                    angle, 1.0))
    else:
        out = bgr.copy()
        M = None

    if crop_to_text:
        out = _crop_to_text(out, mask, scale, bgr.shape[:2], angle if applied else 0.0,
                            margin)

    info = {
        "angle": round(float(angle), 3),
        # The other rotation that also levels the text, 180 deg from this one.
        # Geometry cannot choose between them -- a projection profile of
        # upside-down writing is identical -- so the OCR stage decides, by
        # reading both ways and keeping the higher-confidence result.
        "angle_alt": round(float(angle + 180.0 if angle < 0 else angle - 180.0), 3),
        "confidence": round(float(conf), 3),
        "method": method,
        "applied": bool(applied),
        "ink_ratio": round(ink_ratio, 6),
        "candidates": {k: (None if v is None else round(float(v), 3))
                       for k, v in cands.items()},
        "n_components": dbg.get("n_comp", 0),
    }
    if return_debug:
        dbg["mask"] = mask
        dbg["scale"] = scale
        dbg["border"] = border
        return out, info, dbg
    return out, info


def _crop_to_text(out, mask, scale, in_shape, angle, margin=0.12):
    """Trim to the writing, carrying the ink mask we already have through the
    same rotation.

    Re-detecting ink on the rotated image instead looks simpler and is wrong:
    that image now has constant-fill wedges whose edges answer the detector, so
    the box grows to the whole frame and nothing gets cropped.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return out
    H, W = in_shape                       # full-res input
    p = np.stack([xs / scale, ys / scale], 1).astype(np.float64)

    t = math.radians(angle)
    R = np.array([[math.cos(t), math.sin(t)],
                  [-math.sin(t), math.cos(t)]])
    nh, nw = out.shape[:2]
    q = (p - np.array([W / 2.0, H / 2.0])) @ R.T + np.array([nw / 2.0, nh / 2.0])

    x0, y0 = q.min(0)
    x1, y1 = q.max(0)
    pw, ph = (x1 - x0 + 1) * margin, (y1 - y0 + 1) * margin
    x0 = int(max(0, x0 - pw)); y0 = int(max(0, y0 - ph))
    x1 = int(min(nw, x1 + pw)); y1 = int(min(nh, y1 + ph))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return out
    return out[y0:y1, x0:x1]


# ----------------------------------------------------------------------------
# debug panel
# ----------------------------------------------------------------------------

def debug_panel(orig, out, info, dbg, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 4, figsize=(20, 5.2))
    ax[0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)); ax[0].set_title("input")
    ax[1].imshow(dbg["mask"], cmap="gray"); ax[1].set_title(
        f"ink mask ({info['n_components']} comps)")
    a, s = dbg["sweep_angles"], dbg["sweep_scores"]
    ax[2].plot(a, s, lw=1)
    ax[2].axvline(info["angle"], color="r", ls="--")
    ax[2].set_title(f"sweep -> {info['angle']:+.2f} deg  (conf {info['confidence']:.2f})")
    ax[2].set_xlabel("candidate rotation (deg)")
    ax[3].imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)); ax[3].set_title("output")
    for k in (0, 1, 3):
        ax[k].axis("off")
    fig.suptitle(title or f"{info['method']}  {info['candidates']}", fontsize=9)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def _run_one(path, outpath, debugpath=None, crop=False):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"  !! cannot read {path}", file=sys.stderr)
        return None
    out, info, dbg = fix_rotation(img, crop_to_text=crop, return_debug=True)
    if outpath:
        cv2.imwrite(outpath, out)
    if debugpath:
        fig = debug_panel(img, out, info, dbg, title=os.path.basename(path))
        fig.savefig(debugpath, dpi=90)
        import matplotlib.pyplot as plt
        plt.close(fig)
    print(f"  {os.path.basename(path):38s} angle={info['angle']:+7.2f}  "
          f"conf={info['confidence']:.2f}  comps={info['n_components']}  "
          f"alts={info['candidates']}")
    return info


def main():
    ap = argparse.ArgumentParser(description="Straighten tilted text in a photo.")
    ap.add_argument("input", help="image file or folder")
    ap.add_argument("-o", "--output", default=None, help="output file or folder")
    ap.add_argument("--debug", default=None, help="debug panel png (single file mode)")
    ap.add_argument("--debug-dir", default=None, help="debug panels folder")
    ap.add_argument("--crop", action="store_true", help="crop to the text block")
    a = ap.parse_args()

    if os.path.isdir(a.input):
        outdir = a.output or (a.input.rstrip("/") + "_rot")
        os.makedirs(outdir, exist_ok=True)
        if a.debug_dir:
            os.makedirs(a.debug_dir, exist_ok=True)
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
        for f in sorted(os.listdir(a.input)):
            if not f.lower().endswith(exts):
                continue
            stem = os.path.splitext(f)[0]
            _run_one(os.path.join(a.input, f),
                     os.path.join(outdir, stem + "_rot.jpg"),
                     os.path.join(a.debug_dir, stem + "_dbg.png") if a.debug_dir else None,
                     a.crop)
    else:
        out = a.output or (os.path.splitext(a.input)[0] + "_rot.jpg")
        _run_one(a.input, out, a.debug, a.crop)


if __name__ == "__main__":
    main()
