#!/usr/bin/env python3
"""Capture-quality check for a cropped text line: is it shaken (motion blur) or too small to read?

Why: restoration cannot recover text smeared by camera shake (see docs/SUPER_RESOLUTION_RUNS.md), but a
retake can. The app should tell the user to retake the photo, or to move closer, BEFORE running OCR.

Two measurements on the line crop, both taken after scaling the crop to a fixed 64 px height so crops of
different size compare fairly (the same convention as make_airdrop_test_pairs.sharpness):

  sharpness    variance of the Laplacian of the board-normalised grey image: low = soft strokes
  coherence    how strongly the image gradients share ONE direction (structure tensor, 0..1). Handwriting
               has strokes in every direction; shake smears every stroke along the shake direction, so
               the remaining edges all point the same way.

Thresholds are calibrated by `python capture_check.py --calibrate` on the team's real camera-shake crops
(PhotosToUse/*blur.jpg) against every other line crop we have (Airdrop, 17 Oct, PhotosToUse clear
variants), and saved to capture_check_thresholds.json next to this script.

"Too small" is judged by the TEXT HEIGHT: the height of the line's ink band (the rows that hold most of the
ink), measured in pixels of the ORIGINAL photo. It is not the crop height: a detected line crop carries
padding (the far B3 crops are 21-39 px tall while their writing is 5-22 px). The threshold,
MIN_TEXT_HEIGHT_PX = 12, rests on two pieces of evidence (docs/SUPER_RESOLUTION_RUNS.md, section 13;
reproduce with `python capture_check.py --calibrate-size`):
  - on ~1,000 line crops cut from the original photos, PaddleOCR's CER rises steadily as the text shrinks and
    is about double below 12 px (Airdrop 0.27 vs 0.15 for the rest, 17 Oct 0.21 vs 0.12);
  - on the far board B3 (word-swap test), lines under 12 px lose +0.148 CER each to blur, the others +0.035.

Use in the app:
    from capture_check import capture_advice, photo_advice
    advice = capture_advice(line_bgr, scale=s)   # s = crop pixels per ORIGINAL photo pixel (1.0 if not enlarged)
    # -> "ok" | "retake: camera shake" | "move closer: text too small"
    message = photo_advice([advice for every line])   # one message for the whole photo
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pair_data  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
THRESHOLDS = Path(__file__).with_name("capture_check_thresholds.json")
WORK_HEIGHT = 64
# Text (ink band) shorter than this, in ORIGINAL photo pixels, is flagged "move closer" (see the docstring).
# Replaces the earlier uncalibrated rule "line crop < 30 px", which measured the crop's padding as well.
MIN_TEXT_HEIGHT_PX = 12.0
# The photo-level message appears when at least this share of its lines is too small (a UI choice, not
# calibrated: one small line in a far corner should not stop the user).
MIN_SHARE_TOO_SMALL = 0.25


def features(bgr: np.ndarray) -> dict:
    grey = pair_data.normalise_board(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).mean(axis=2).astype(np.float32)
    h, w = grey.shape
    grey = cv2.resize(grey, (max(8, round(w * WORK_HEIGHT / h)), WORK_HEIGHT), interpolation=cv2.INTER_AREA)
    sharp = float(cv2.Laplacian(grey, cv2.CV_32F).var())
    smooth = cv2.GaussianBlur(grey, (0, 0), 1.0)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    energy = gx * gx + gy * gy
    strong = energy > np.percentile(energy, 80)          # edges only; the flat board carries no direction
    jxx, jyy, jxy = (float(v[strong].mean()) for v in (gx * gx, gy * gy, gx * gy))
    root = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
    coherence = float(root / (jxx + jyy + 1e-9))
    angle = float(0.5 * np.degrees(np.arctan2(2 * jxy, jxx - jyy)))   # dominant gradient direction
    return {"sharpness": sharp, "coherence": coherence, "gradient_angle": angle}


def text_height_px(bgr: np.ndarray) -> float:
    """Height of the line's ink band in the crop's own pixels (NaN if there is no clear ink).
    Same rule as the word boxes of finetune_tpgsr_whiteboard.word_boxes: ink = clearly darker than the board
    (threshold adapted to the crop, so pale far photos work); long horizontal rules (underlines, table lines)
    are removed first; then the block of rows holding the most ink is kept (not the single inkiest row, which
    can be a slanted underline), and the band = the rows around its peak that hold more than 15% of its ink, so
    fragments of the lines above / below are ignored. Measured on the unsmoothed rows, so it does not depend on
    the crop's resolution."""
    norm = pair_data.normalise_board(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).mean(axis=2)
    dark = float(np.percentile(norm, 1.0))
    if dark > 0.85:                                     # no real ink
        return float("nan")
    ink = (norm < 1.0 - 0.45 * (1.0 - dark)).astype(np.uint8)
    # (no speck filter: at the photo's own resolution far writing has 1 px strokes, which a 2 x 2 opening deletes;
    #  specks are too small to pass the 15% row rule below anyway)
    # Underlines and table rules are the inkiest rows of many crops and would be taken for the text band:
    # remove horizontal runs longer than a fifth of the crop width (no letter stroke is that long).
    rule = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, max(10, ink.shape[1] // 5)), np.uint8))
    ink = ink & (1 - cv2.dilate(rule, np.ones((3, 3), np.uint8)))
    rows = cv2.GaussianBlur(ink.sum(axis=1).astype(np.float32)[:, None], (1, 0), sigmaX=1, sigmaY=3).ravel()
    if rows.max() <= 0:
        return float("nan")
    # 1. blocks of rows with any real ink; keep the block holding the MOST ink in total. (The single inkiest
    #    row can be a slanted underline, which the rule removal above misses: one long row beats any letter row,
    #    but the letters hold far more ink in total.)
    on = np.r_[False, rows > 0.05 * rows.max(), False]
    edges = np.flatnonzero(on[1:] != on[:-1])
    blocks = list(zip(edges[::2], edges[1::2]))
    a, b = max(blocks, key=lambda ab: rows[ab[0]:ab[1]].sum())
    # 2. inside it, the text band = the rows whose UNSMOOTHED ink exceeds 15% of the block's peak row.
    #    (The smoothing only finds the block; measuring on it would widen the band by a fixed number of rows,
    #    which inflates small text and makes the result depend on the crop's resolution.)
    #    Contiguous rows only, and of several runs the one with the most ink: an underline touching the text
    #    through a descender ('g', 'y') is its own, lighter run and is left out.
    raw = ink.sum(axis=1).astype(np.float32)[a:b]
    on = np.r_[False, raw > 0.15 * raw.max(), False]
    edges = np.flatnonzero(on[1:] != on[:-1])
    runs = list(zip(edges[::2], edges[1::2]))
    if not runs:
        return float("nan")
    r0, r1 = max(runs, key=lambda ab: raw[ab[0]:ab[1]].sum())
    return float(r1 - r0)


def load_thresholds() -> dict:
    return json.loads(THRESHOLDS.read_text())


def capture_advice(line_bgr: np.ndarray, line_height_px: float | None = None, thresholds: dict | None = None,
                   scale: float | None = None) -> str:
    """'retake: camera shake', 'move closer: text too small' or 'ok' for one cropped text line.

    The size check needs to know how the crop relates to the ORIGINAL photo, given either way:
      scale           crop pixels per original-photo pixel (1.0 = cropped from the photo, 3.0 = enlarged 3x)
      line_height_px  the crop's height in the original photo (older interface; converted to a scale)
    With neither, only the shake check runs (the size of an already enlarged crop says nothing)."""
    t = thresholds or load_thresholds()
    f = features(line_bgr)
    if f["coherence"] >= t["coherence_min"] and f["sharpness"] <= t["sharpness_max"]:
        return "retake: camera shake"
    if scale is None and line_height_px:
        scale = line_bgr.shape[0] / float(line_height_px)
    if scale:
        h = text_height_px(line_bgr) / scale
        if h == h and h < MIN_TEXT_HEIGHT_PX:           # h == h: skip crops with no measurable ink
            return "move closer: text too small"
    return "ok"


def photo_advice(line_advices: list[str], min_share: float = MIN_SHARE_TOO_SMALL) -> str:
    """One message for the whole photo from its per-line advice."""
    n = len(line_advices)
    if not n:
        return "ok"
    if sum(a.startswith("retake") for a in line_advices) / n >= min_share:
        return "retake: camera shake (hold the phone still)"
    if sum(a.startswith("move closer") for a in line_advices) / n >= min_share:
        return "move closer: the writing is too small to read reliably"
    return "ok"


# ------------------------------------------------------------------------------------------------
def crop_sets() -> dict[str, list[Path]]:
    photos = REPO.parent / "PhotosToUse"
    sets = {"shake (real)": sorted(photos.glob("*blur.jpg")),
            "PhotosToUse clear variants": sorted(p for p in photos.iterdir()
                                                 if not p.name[0].isdigit() and "blur" not in p.name)}
    for name, folder in (("Airdrop line crops", REPO.parent / "airdrop_rec_paddle"),
                         ("17 Oct line crops", REPO.parent / "17oct_rec_paddle")):
        with (folder / "manifest.csv").open(newline="", encoding="utf-8") as fh:
            sets[name] = [folder / r["crop"] for r in csv.DictReader(fh)]
    return sets


def fit_thresholds(pos: list[dict], neg: list[dict]) -> tuple[float, float]:
    """Rule: shaken = coherence >= c AND sharpness <= s. c and s are placed so every shake crop in `pos`
    is caught, each halfway into the gap to the nearest non-shaken crop (a margin, not the edge)."""
    c_min = min(r["coherence"] for r in pos)
    s_max = max(r["sharpness"] for r in pos)
    below = [r["coherence"] for r in neg if r["sharpness"] <= s_max and r["coherence"] < c_min]
    c_thr = (c_min + max(below)) / 2 if below else c_min
    above = [r["sharpness"] for r in neg if r["coherence"] >= c_thr and r["sharpness"] > s_max]
    s_thr = (s_max + min(above)) / 2 if above else s_max * 1.25
    return c_thr, s_thr


def calibrate() -> None:
    rows = []
    for set_name, paths in crop_sets().items():
        for p in paths:
            img = cv2.imread(str(p))
            if img is None or min(img.shape[:2]) < 8:
                continue
            rows.append({"set": set_name, "file": p.name, **features(img)})
    pos = [r for r in rows if r["set"] == "shake (real)"]
    neg = [r for r in rows if r["set"] != "shake (real)"]
    c_thr, s_thr = fit_thresholds(pos, neg)
    thresholds = {"coherence_min": round(c_thr, 4), "sharpness_max": round(s_thr, 6),
                  "calibrated_on": {k: sum(r["set"] == k for r in rows) for k in sorted({r["set"] for r in rows})}}
    flagged = lambda r: r["coherence"] >= c_thr and r["sharpness"] <= s_thr
    print(f"rule: shaken if coherence >= {c_thr:.3f} and sharpness <= {s_thr:.5f}")
    for set_name in sorted({r["set"] for r in rows}):
        part = [r for r in rows if r["set"] == set_name]
        n = sum(flagged(r) for r in part)
        coh = np.percentile([r["coherence"] for r in part], [5, 50, 95])
        sh = np.percentile([r["sharpness"] for r in part], [5, 50, 95])
        print(f"  {set_name:<28} {len(part):>4} crops | flagged {n:>3} | coherence p5/50/95 "
              f"{coh[0]:.2f}/{coh[1]:.2f}/{coh[2]:.2f} | sharpness p5/50/95 {sh[0]:.4f}/{sh[1]:.4f}/{sh[2]:.4f}")
    false_alarms = [r for r in neg if flagged(r)]
    for r in false_alarms[:20]:
        print(f"    false alarm: {r['set']}: {r['file']} (coherence {r['coherence']:.2f}, sharpness {r['sharpness']:.5f})")
    # leave-one-out on the shake crops: would each one be caught by thresholds fitted WITHOUT it?
    caught = 0
    for i in range(len(pos)):
        c_i, s_i = fit_thresholds(pos[:i] + pos[i + 1:], neg)
        caught += pos[i]["coherence"] >= c_i and pos[i]["sharpness"] <= s_i
    print(f"  leave-one-out: {caught}/{len(pos)} shake crops still caught when thresholds are set without them")
    thresholds["false_alarms"] = len(false_alarms)
    thresholds["leave_one_out_caught"] = f"{caught}/{len(pos)}"
    THRESHOLDS.write_text(json.dumps(thresholds, indent=1) + "\n")
    report = REPO / "outputs" / "capture_check_calibration.csv"
    with report.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) + ["flagged"])
        w.writeheader()
        w.writerows({**r, "flagged": flagged(r)} for r in rows)
    print(f"saved {THRESHOLDS.name} and {report}")


def calibrate_size() -> None:
    """Evidence for MIN_TEXT_HEIGHT_PX, in the app's own units (text height measured by text_height_px on
    crops at the original photo's resolution):
      1. ~1,000 line crops cut from the original photos (airdrop_rec_paddle, 17oct_rec_paddle) with the CER
         of PaddleOCR's reading stored in their manifests: where does reading start to break down?
      2. the far lines of the word-swap test (20 Oct): how much would a photo as close as the clear one win?
         These are the SR TEST lines, so they support the choice but must not be used to fine-tune it."""
    import finetune_tpgsr_whiteboard as ft
    bins = [(0, 8), (8, 10), (10, 12), (12, 14), (14, 16), (16, 20), (20, 30), (30, 10 ** 6)]
    print("1. PaddleOCR CER of line crops by their text height (original photo pixels)")
    rec_rows = []
    for name, folder in (("Airdrop", REPO.parent / "airdrop_rec_paddle"), ("17 Oct", REPO.parent / "17oct_rec_paddle")):
        man = folder / "manifest.csv"
        if not man.exists():
            print(f"  missing {man}: skipped")
            continue
        for r in csv.DictReader(man.open(encoding="utf-8")):
            img = cv2.imread(str(folder / r["crop"]))
            if img is None or min(img.shape[:2]) < 8 or r.get("cer") in (None, ""):
                continue
            rec_rows.append(dict(set=name, crop=r["crop"], condition=r.get("condition", ""),
                                 text_height_px=text_height_px(img), cer=float(r["cer"])))
    for name in dict.fromkeys(r["set"] for r in rec_rows):
        part = [r for r in rec_rows if r["set"] == name]
        h, c = np.array([r["text_height_px"] for r in part]), np.array([r["cer"] for r in part])
        print(f"  {name}: {len(part)} crops, text height median {np.nanmedian(h):.0f} px")
        for lo, hi in bins:
            m = (h >= lo) & (h < hi)
            if m.any():
                print(f"    {lo:>3}-{'' if hi > 999 else hi:<3} px: {m.sum():>4} crops | mean CER {c[m].mean():.2f}")
        f = h < MIN_TEXT_HEIGHT_PX
        print(f"    -> flagged by MIN_TEXT_HEIGHT_PX = {MIN_TEXT_HEIGHT_PX:g}: {f.sum()} crops ({f.mean():.1%}), "
              f"CER {c[f].mean():.2f} vs {c[~f].mean():.2f} for the rest")

    swap = REPO / "outputs" / "tpgsr_whiteboard" / "word_swap_lines.csv"
    if not swap.exists():
        print(f"\n2. skipped: missing {swap.relative_to(REPO)} (run scripts/word_swap_counterfactual.py)")
        return
    spp = {}
    for path in ft.MANIFESTS.values():
        for r in csv.DictReader(path.open(encoding="utf-8")):
            try:
                spp[r["pair_id"]] = float(r["source_pixels_per_target_pixel"])
            except (KeyError, ValueError):
                pass
    lines = {ln["pair_id"]: ln for ln in json.loads((ft.OUT / "lines.json").read_text())["lines"]}
    rows = []
    for r in csv.DictReader(swap.open(encoding="utf-8")):
        ln, k = lines[r["pair_id"]], spp.get(r["pair_id"])
        if not k:
            continue
        big = cv2.imread(str(REPO / ln["input_path"]))          # far crop, enlarged by 1/k to the target size
        img = cv2.resize(big, (max(8, round(big.shape[1] * k)), max(8, round(big.shape[0] * k))), interpolation=cv2.INTER_AREA)
        rows.append(dict(set=r["set"], pair_id=r["pair_id"], label=r["label"], text_height_px=round(text_height_px(img), 1),
                         crop_height_px=img.shape[0], raw_cer=float(r["raw_cer"]), clear_words_cer=float(r["all_clear_cer"])))
    print("\n2. Word-swap far lines: blur cost = raw CER - CER with every word from the clear photo (SR test lines)")
    for set_name in dict.fromkeys(r["set"] for r in rows):
        part = [r for r in rows if r["set"] == set_name]
        h = np.array([r["text_height_px"] for r in part])
        gap = np.array([r["raw_cer"] - r["clear_words_cer"] for r in part])
        f = h < MIN_TEXT_HEIGHT_PX
        print(f"  {set_name}: {len(part)} lines, text height median {np.nanmedian(h):.1f} px (range {np.nanmin(h):.0f}-"
              f"{np.nanmax(h):.0f}); crop height median {np.median([r['crop_height_px'] for r in part]):.0f} px")
        if f.any():
            print(f"    blur cost of the {f.sum()} lines under {MIN_TEXT_HEIGHT_PX:g} px: {gap[f].mean():+.3f} | "
                  f"of the other {(~f).sum()}: {gap[~f].mean():+.3f}")
        for t in (8, 10, 12, 16, 20, 25):
            ft_ = h < t
            cer = np.mean([r["clear_words_cer"] if fl else r["raw_cer"] for r, fl in zip(part, ft_)])
            mark = "  <- MIN_TEXT_HEIGHT_PX" if t == MIN_TEXT_HEIGHT_PX else ""
            print(f"    flag < {t:>2} px: {ft_.sum():>2} of {len(part)} lines | mean CER if they were retaken as close as the clear "
                  f"photo {cer:.3f} (as photographed {np.mean([r['raw_cer'] for r in part]):.3f}){mark}")
    out = REPO / "outputs" / "capture_size_calibration.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["source", "set", "id", "text_height_px", "cer", "clear_words_cer"])
        w.writeheader()
        w.writerows(dict(source="line crops", set=r["set"], id=r["crop"], text_height_px=r["text_height_px"], cer=r["cer"],
                         clear_words_cer="") for r in rec_rows)
        w.writerows(dict(source="word-swap far lines", set=r["set"], id=r["pair_id"], text_height_px=r["text_height_px"],
                         cer=r["raw_cer"], clear_words_cer=r["clear_words_cer"]) for r in rows)
    print(f"\nsaved {out.relative_to(REPO)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--calibrate-size", action="store_true", help="evidence for the 'move closer' threshold")
    ap.add_argument("--scale", type=float, default=1.0, help="with an image: crop pixels per original photo pixel")
    ap.add_argument("image", nargs="?", type=Path)
    a = ap.parse_args()
    if a.calibrate:
        calibrate()
    elif a.calibrate_size:
        calibrate_size()
    elif a.image:
        img = cv2.imread(str(a.image))
        print(features(img), f"text height {text_height_px(img) / a.scale:.1f} px", capture_advice(img, scale=a.scale))
    else:
        ap.print_help()
