#!/usr/bin/env python3
"""Synthetic TRAINING pairs from sharp Airdrop line crops of boards that are NOT test boards.

Why: the 17 Oct pairs come from one board, one writer and one blue marker, and the restorer learned that
marker's look instead of sharper strokes. These pairs add other boards, handwriting and ink (black and
blue markers on the quiz, lecture-notes and animal boards).

The Airdrop boards used as the cross-board TEST (television, CNN = the group_ids in
paired_dataset_airdrop/airdrop_test_manifest.csv) are excluded, so the test stays on unseen boards.

For each line: target = its sharpest crop at least TARGET_MIN_H px tall (same choice as the test pairs).
Each target gets VARIANTS degraded copies made with the team's own recipe (prepare_real_pairs.synthetic_degrade:
blur, downscale, sensor noise, JPEG, bicubic back up), except that the downscale factor is 3-6x instead of
2-4x, because real far / side photos needed 3.3-6.7x enlargement (source_pixels_per_target_pixel 0.15-0.30).

Output: outputs/paired_dataset_airdrop_synth/{targets,inputs_synthetic}/ and airdrop_synth_manifest.csv
(same columns as the 17 Oct accepted_manifest.csv; pair_type = synthetic, split = train).
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
from make_airdrop_test_pairs import CROPS, REPO, TARGET_MIN_H, TEMPLATE, read_rgb, sharpness  # noqa: E402

OUT = REPO / "outputs" / "paired_dataset_airdrop_synth"
TEST_MANIFEST = REPO / "outputs" / "paired_dataset_airdrop" / "airdrop_test_manifest.csv"
VARIANTS = 4
FACTORS, FACTOR_P = [3, 4, 6], [0.35, 0.40, 0.25]


def synthetic_degrade(target: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """prepare_real_pairs.synthetic_degrade with downscale factors 3-6x (see module docstring)."""
    factor = int(rng.choice(FACTORS, p=FACTOR_P))
    sigma = float(rng.uniform(0.4, 1.5))
    work = cv2.GaussianBlur(target, (0, 0), sigma)
    h, w = work.shape[:2]
    low = cv2.resize(work, (max(2, w // factor), max(2, h // factor)), interpolation=cv2.INTER_AREA)
    noise_std = float(rng.uniform(0.0, 4.0))
    if noise_std:
        low = np.clip(low.astype(np.float32) + rng.normal(0, noise_std, low.shape), 0, 255).astype(np.uint8)
    jpeg_quality = int(rng.integers(50, 91))
    ok, encoded = cv2.imencode(".jpg", low, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if ok:
        low = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    degraded = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
    return degraded, {"downscale": factor, "blur_sigma": round(sigma, 3),
                      "noise_std": round(noise_std, 3), "jpeg_quality": jpeg_quality}


def main() -> None:
    with TEMPLATE.open(newline="", encoding="utf-8") as fh:
        columns = list(csv.DictReader(fh).fieldnames)
    with TEST_MANIFEST.open(newline="", encoding="utf-8") as fh:
        test_boards = {r["group_id"].replace("airdrop_", "") for r in csv.DictReader(fh)}
    with (CROPS / "manifest.csv").open(newline="", encoding="utf-8") as fh:
        crops = [r for r in csv.DictReader(fh) if r["status"] in ("accepted", "duplicate")
                 and r["board"] not in test_boards]
    groups = defaultdict(list)
    for r in crops:
        version = r["board_version"] if r["board"] == "B4" else ""
        groups[(r["board"], version, r["label"])].append(r)

    (OUT / "targets").mkdir(parents=True, exist_ok=True)
    (OUT / "inputs_synthetic").mkdir(parents=True, exist_ok=True)
    rows = []
    for (board, version, label), members in sorted(groups.items()):
        big = [r for r in members if int(r["crop_h"]) >= TARGET_MIN_H]
        if not big:
            continue
        images = {r["crop"]: read_rgb(CROPS / r["crop"]) for r in big}
        target = max(big, key=lambda r: sharpness(images[r["crop"]]))
        tgt_bgr = cv2.cvtColor(images[target["crop"]], cv2.COLOR_RGB2BGR)
        tgt_name = f"targets/{target['photo']}_{target['line_id']}.png"
        cv2.imwrite(str(OUT / tgt_name), tgt_bgr)
        seed = int(hashlib.sha1(f"{board}|{version}|{label}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        for k in range(1, VARIANTS + 1):
            degraded, params = synthetic_degrade(tgt_bgr, rng)
            pair_id = f"{target['photo']}_{target['line_id']}__S{k:02d}"
            in_name = f"inputs_synthetic/{pair_id}.png"
            cv2.imwrite(str(OUT / in_name), degraded)
            row = {c: "" for c in columns}
            row.update({
                "pair_id": pair_id, "pair_type": "synthetic", "group_id": f"airdrop_synth_{board}",
                "split": "train", "input_path": str((OUT / in_name).relative_to(REPO)),
                "target_path": str((OUT / tgt_name).relative_to(REPO)),
                "input_photo": f"{target['photo']}.HEIC", "target_photo": f"{target['photo']}.HEIC",
                "input_role": "synthetic_from_close_target", "review_status": "accepted",
                "target_sharpness": round(sharpness(images[target["crop"]]), 2),
                "source_pixels_per_target_pixel": round(1 / params["downscale"], 4),
                "suggested_upscale": params["downscale"],
                "degradation": ";".join(f"{a}={b}" for a, b in params.items()),
                "source_set": "airdrop_synthetic", "transcription_verified": label, "cer_eligible": "yes",
                "annotation_source": "human (Airdrop_Labels.xlsx)",
            })
            rows.append(row)

    with (OUT / "airdrop_synth_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    by = defaultdict(int)
    for r in rows:
        by[r["group_id"]] += 1
    print(f"{len(rows)} synthetic training pairs from {len(rows) // VARIANTS} lines "
          f"(test boards {sorted(test_boards)} excluded) -> {OUT}")
    for g, n in sorted(by.items()):
        print(f"  {g:<20} {n}")


if __name__ == "__main__":
    main()
