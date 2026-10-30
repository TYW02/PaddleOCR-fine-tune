#!/usr/bin/env python3
"""Build a CROSS-BOARD test set of real (far crop -> sharp crop) pairs from the Airdrop line crops.

The 17 Oct pairs all come from one board and one writer. These pairs come from four other boards
(television, CNN, quiz, lecture notes), so they test whether the restorer generalises.

Source: Assignment/airdrop_rec_paddle/manifest.csv -- line crops already cut from perspective-
corrected photos by the team pipeline (notebook 04), each with a human transcription.

For every line (same board; for the quiz board also the same erasure state):
  target = the sharpest crop at least TARGET_MIN_H px tall
  inputs = crops of the same line from OTHER photos that are at most MAX_SPP x the target height
For each input:
  1. enlarge it to the target's height (bicubic) -- the same "enlarge first" step used at inference
  2. align it to the target with an affine ECC fit (crops from two photos never line up exactly)
     (started from the ink bounding boxes, because the two crops have different margins)
  3. keep it only if the fit is sane (scale ~1, small rotation) and the ink correlation is high
     enough -- the same ink_correlation used for the 17 Oct pairs

Output (whiteboard-ocr/outputs/paired_dataset_airdrop/): inputs_real/, targets/, and
airdrop_test_manifest.csv with the same columns as the 17 Oct accepted_manifest.csv, split = test.
review_status: accepted (ink correlation >= ACCEPT), needs_review (>= REVIEW), rejected pairs are
listed in airdrop_rejected.csv. Nothing is written into the Airdrop folders.
"""
from __future__ import annotations

import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pair_data  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CROPS = REPO.parent / "airdrop_rec_paddle"
OUT = REPO / "outputs" / "paired_dataset_airdrop"
TEMPLATE = REPO / "outputs" / "paired_dataset_17oct" / "accepted_manifest.csv"

TARGET_MIN_H = 80      # a usable sharp target is at least this tall (px)
MAX_SPP = 0.40         # input crop height / target height: at least 2.5x enlargement
ACCEPT = 0.60          # ink correlation for an accepted pair (17 Oct accepted pairs: median 0.82)
REVIEW = 0.45          # between REVIEW and ACCEPT -> needs_review
MAX_SCALE_DEV = 0.25   # affine scale (after the ink-box start) must stay within 1 +- this
MAX_ROT_DEG = 6.0


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def sharpness(rgb: np.ndarray) -> float:
    """Stroke sharpness on a copy scaled to 64 px tall, so crops of different size compare fairly."""
    g = pair_data.normalise_board(rgb).mean(axis=2)
    h, w = g.shape
    g = cv2.resize(g, (max(1, round(w * 64 / h)), 64), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def board_colour(rgb: np.ndarray) -> tuple[int, int, int]:
    return tuple(int(v) for v in np.median(rgb.reshape(-1, 3)[::7], axis=0))


def ink_box(gray01: np.ndarray):
    """Robust bounding box (x0, y0, x1, y1) of the strokes in a normalised grey image."""
    ink = cv2.GaussianBlur(gray01, (0, 0), 1.0) < 0.75
    ys, xs = np.nonzero(ink)
    if len(xs) < 20:
        return None
    return (float(np.percentile(xs, 1)), float(np.percentile(ys, 2)),
            float(np.percentile(xs, 99)), float(np.percentile(ys, 98)))


def align(inp_rgb: np.ndarray, tgt_rgb: np.ndarray):
    """Enlarge the input to the target height, place it on a target-size canvas, refine with affine ECC.
    Returns (aligned input RGB, ECC correlation, affine matrix) or (None, reason, None)."""
    th, tw = tgt_rgb.shape[:2]
    ih, iw = inp_rgb.shape[:2]
    s = th / ih
    up = cv2.resize(inp_rgb, (max(1, round(iw * s)), th), interpolation=cv2.INTER_CUBIC)
    fill = board_colour(inp_rgb)
    canvas = np.empty((th, tw, 3), np.uint8)
    canvas[:] = fill
    uw = up.shape[1]
    if uw >= tw:                                   # centre-crop / centre-place horizontally
        x0 = (uw - tw) // 2
        canvas[:] = up[:, x0:x0 + tw]
    else:
        x0 = (tw - uw) // 2
        canvas[:, x0:x0 + uw] = up
    t = pair_data.normalise_board(tgt_rgb).mean(axis=2).astype(np.float32)
    c = pair_data.normalise_board(canvas).mean(axis=2).astype(np.float32)
    # Start from the ink bounding boxes: crops from two photos have different margins around the text,
    # and ECC only refines locally. warp maps target coordinates -> canvas coordinates.
    bt, bc = ink_box(t), ink_box(c)
    if bt is None or bc is None:
        return None, "no ink found", None
    sx = (bc[2] - bc[0]) / max(bt[2] - bt[0], 1.0)
    sy = (bc[3] - bc[1]) / max(bt[3] - bt[1], 1.0)
    warp = np.float32([[sx, 0, bc[0] - sx * bt[0]], [0, sy, bc[1] - sy * bt[1]]])
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-6)
    try:
        cc, warp = cv2.findTransformECC(t, c, warp, cv2.MOTION_AFFINE, criteria, None, 5)
    except cv2.error:
        return None, "ECC did not converge", None
    a = warp[:, :2]
    sx, sy = float(np.linalg.norm(a[:, 0])), float(np.linalg.norm(a[:, 1]))
    rot = float(np.degrees(np.arctan2(a[1, 0], a[0, 0])))
    # Both crops come from rectified photos: after the ink-box start the fit should be close to 1:1
    # and hardly rotated. A large residual scale or rotation means ECC locked onto the wrong strokes.
    if abs(sx - 1) > MAX_SCALE_DEV or abs(sy - 1) > MAX_SCALE_DEV or abs(rot) > MAX_ROT_DEG:
        return None, f"implausible fit (scale {sx:.2f}/{sy:.2f}, rotation {rot:.1f} deg)", None
    aligned = cv2.warpAffine(canvas, warp, (tw, th), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
    return aligned, float(cc), warp


def main() -> None:
    with TEMPLATE.open(newline="", encoding="utf-8") as fh:
        columns = list(csv.DictReader(fh).fieldnames)
    with (CROPS / "manifest.csv").open(newline="", encoding="utf-8") as fh:
        crops = [r for r in csv.DictReader(fh) if r["status"] in ("accepted", "duplicate")]
    groups = defaultdict(list)
    for r in crops:
        version = r["board_version"] if r["board"] == "B4" else ""
        groups[(r["board"], version, r["label"])].append(r)

    (OUT / "inputs_real").mkdir(parents=True, exist_ok=True)
    (OUT / "targets").mkdir(parents=True, exist_ok=True)
    rows, rejected = [], []
    for (board, version, label), members in sorted(groups.items()):
        big = [r for r in members if int(r["crop_h"]) >= TARGET_MIN_H]
        if not big:
            continue
        images = {r["crop"]: read_rgb(CROPS / r["crop"]) for r in members}
        target = max(big, key=lambda r: sharpness(images[r["crop"]]))
        th = int(target["crop_h"])
        inputs = [r for r in members if r["photo"] != target["photo"] and int(r["crop_h"]) <= MAX_SPP * th]
        if not inputs:
            continue
        tgt_rgb = images[target["crop"]]
        tgt_name = f"targets/{target['photo']}_{target['line_id']}.png"
        for r in inputs:
            aligned, info, _ = align(images[r["crop"]], tgt_rgb)
            spp = int(r["crop_h"]) / th
            base = {"board": board, "label": label, "input": r["crop"], "target": target["crop"], "spp": round(spp, 3)}
            if aligned is None:
                rejected.append({**base, "reason": info}); continue
            ok = pair_data.valid_mask(aligned)
            if ok.mean() < 0.5:
                rejected.append({**base, "reason": f"only {ok.mean():.0%} of the aligned input is real pixels"})
                continue
            ys, xs = np.nonzero(ok)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            inkc = pair_data.ink_correlation(aligned[y0:y1, x0:x1], tgt_rgb[y0:y1, x0:x1])
            if inkc < REVIEW:
                rejected.append({**base, "reason": f"ink correlation {inkc:.2f} < {REVIEW}"}); continue
            status = "accepted" if inkc >= ACCEPT else "needs_review"
            digest = hashlib.sha1(f"{r['crop']}|{target['crop']}".encode()).hexdigest()[:8]
            pair_id = f"{r['photo']}__{target['photo']}__{board}_{r['line_id']}_{digest}"
            in_name = f"inputs_real/{pair_id}.png"
            cv2.imwrite(str(OUT / in_name), cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR))
            if not (OUT / tgt_name).exists():
                cv2.imwrite(str(OUT / tgt_name), cv2.cvtColor(tgt_rgb, cv2.COLOR_RGB2BGR))
            row = {c: "" for c in columns}
            row.update({
                "pair_id": pair_id, "pair_type": "real", "group_id": f"airdrop_{board}", "split": "test",
                "input_path": str((OUT / in_name).relative_to(REPO)),
                "target_path": str((OUT / tgt_name).relative_to(REPO)),
                "input_photo": f"{r['photo']}.HEIC", "target_photo": f"{target['photo']}.HEIC",
                "input_role": r["condition"], "transcription_draft": r["ocr_reading"],
                "recognition_confidence": r["ocr_conf"], "review_status": status,
                "alignment_inlier_ratio": round(info, 4), "ink_correlation": round(inkc, 4),
                "target_sharpness": round(sharpness(tgt_rgb), 2),
                "input_sharpness": round(sharpness(images[r["crop"]]), 2),
                "source_pixels_per_target_pixel": round(spp, 4), "suggested_upscale": round(1 / spp, 2),
                "degradation": "real_camera", "source_set": "airdrop",
                "transcription_verified": label, "cer_eligible": "yes",
                "review_notes": "alignment_inlier_ratio holds the ECC correlation of the affine fit",
                "annotation_source": "human (Airdrop_Labels.xlsx)",
            })
            rows.append(row)

    def write(path, data, fields):
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(data)

    write(OUT / "airdrop_test_manifest.csv", rows, columns)
    write(OUT / "airdrop_rejected.csv", rejected, ["board", "label", "input", "target", "spp", "reason"])
    acc = sum(r["review_status"] == "accepted" for r in rows)
    print(f"{len(rows)} pairs written ({acc} accepted, {len(rows) - acc} needs_review), "
          f"{len(rejected)} rejected -> {OUT}")
    by = defaultdict(int)
    for r in rows:
        by[(r["group_id"], r["review_status"])] += 1
    for k, v in sorted(by.items()):
        print(f"  {k[0]:<12} {k[1]:<13} {v}")


if __name__ == "__main__":
    main()
