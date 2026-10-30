#!/usr/bin/env python3
"""Select bicubic or blended TPGSR output using PaddleOCR confidence.

The confidence margin is selected only on the untouched calibration split. The
TextZoom hard test split is evaluated once after the margin has been fixed.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/whiteboard-ocr-matplotlib")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from paddleocr import TextRecognition
from torch.utils.data import ConcatDataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_tpgsr_blend import blend, generate_images, scores
from train_tpgsr_textzoom import (
    Settings,
    TextZoomPairs,
    build_generator,
    build_recognizer,
    choose_device,
    project_paths,
)


# A margin of 1.01 is the explicit "never use SR" baseline because PaddleOCR
# confidence is bounded by 1.0.
MARGINS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 1.01]


def recognise_with_confidence(
    model: TextRecognition,
    images: list[np.ndarray],
    chunk_size: int = 128,
) -> tuple[list[str], list[float]]:
    texts: list[str] = []
    confidences: list[float] = []
    for start in range(0, len(images), chunk_size):
        batch = [
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            for image in images[start:start + chunk_size]
        ]
        predictions = model.predict(batch)
        for prediction in predictions:
            texts.append(str(prediction["rec_text"]))
            confidences.append(float(prediction["rec_score"]))
        print(f"  PaddleOCR {min(start + chunk_size, len(images))}/{len(images)}")
    return texts, confidences


def gated_predictions(
    bicubic_text: list[str],
    bicubic_confidence: list[float],
    blend_text: list[str],
    blend_confidence: list[float],
    margin: float,
) -> tuple[list[str], np.ndarray]:
    use_blend = (
        np.asarray(blend_confidence) > np.asarray(bicubic_confidence) + margin
    )
    selected = [
        sr if choose_sr else base
        for base, sr, choose_sr in zip(bicubic_text, blend_text, use_blend)
    ]
    return selected, use_blend


def main() -> None:
    _, source, textzoom, output = project_paths()
    checkpoint = output / "tpgsr_soft_prior_best.pt"
    blend_summary_path = output / "blend_summary.json"
    if not checkpoint.exists() or not blend_summary_path.exists():
        raise FileNotFoundError("Run the soft-prior and blend experiments first")

    blend_summary = json.loads(blend_summary_path.read_text())
    alpha = float(blend_summary["selected_alpha"])
    device = choose_device()
    recognizer = build_recognizer(source, device)
    generator = build_generator(source, device)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    generator.load_state_dict(saved["generator"])
    generator.eval()
    paddle = TextRecognition(model_name="PP-OCRv6_medium_rec")
    print(
        f"device: {device}; checkpoint epoch: {saved.get('epoch')}; "
        f"fixed blend alpha: {alpha:.2f}"
    )

    # This is the same untouched calibration split used to choose blend alpha.
    # It is still separate from the official TextZoom hard test split.
    full_train = ConcatDataset([
        TextZoomPairs(textzoom / "train1"),
        TextZoomPairs(textzoom / "train2"),
    ])
    rng = np.random.default_rng(Settings().seed)
    order = rng.permutation(len(full_train))
    calibration_indices = order[512:1000]
    print(f"Generating {len(calibration_indices)} calibration pairs...")
    calibration_labels, calibration_bicubic, calibration_neural, _ = generate_images(
        full_train, calibration_indices, generator, recognizer, device
    )
    calibration_blend = blend(calibration_bicubic, calibration_neural, alpha)

    print("Calibration OCR: bicubic")
    cal_base_text, cal_base_conf = recognise_with_confidence(paddle, calibration_bicubic)
    print("Calibration OCR: blend")
    cal_blend_text, cal_blend_conf = recognise_with_confidence(paddle, calibration_blend)
    base_cer, base_exact, _, _ = scores(calibration_labels, cal_base_text)

    calibration_rows: list[dict] = []
    for margin in MARGINS:
        selected_text, selected_mask = gated_predictions(
            cal_base_text, cal_base_conf, cal_blend_text, cal_blend_conf, margin
        )
        mean_cer, exact, _, _ = scores(calibration_labels, selected_text)
        calibration_rows.append({
            "margin": margin,
            "cer": mean_cer,
            "exact": exact,
            "selected_blend": int(selected_mask.sum()),
            "selected_fraction": float(selected_mask.mean()),
        })
        print(
            f"margin={margin:.3f}: CER={mean_cer:.4f}; exact={exact:.4f}; "
            f"blend={selected_mask.sum()}/{len(selected_mask)}"
        )

    # Preserve exact-word accuracy on calibration data. If no learned gate can
    # do so, the never-use-SR row remains an eligible safe fallback.
    eligible = [
        row for row in calibration_rows
        if row["exact"] >= base_exact - 1e-12
    ]
    best = min(
        eligible,
        key=lambda row: (
            row["cer"],
            -row["exact"],
            row["selected_blend"],
        ),
    )
    best_margin = float(best["margin"])
    calibration = pd.DataFrame(calibration_rows)
    calibration_path = output / "confidence_gate_calibration.csv"
    calibration.to_csv(calibration_path, index=False)
    print(
        f"Selected margin={best_margin:.3f} using calibration data only "
        f"(bicubic exact={base_exact:.4f})"
    )

    hard = TextZoomPairs(textzoom / "test" / "hard")
    test_indices = np.arange(len(hard))
    print(f"Generating {len(test_indices):,} untouched hard-test pairs...")
    test_labels, test_bicubic, test_neural, test_hr = generate_images(
        hard, test_indices, generator, recognizer, device
    )
    test_blend = blend(test_bicubic, test_neural, alpha)
    print("Test OCR: bicubic")
    test_base_text, test_base_conf = recognise_with_confidence(paddle, test_bicubic)
    print("Test OCR: blend")
    test_blend_text, test_blend_conf = recognise_with_confidence(paddle, test_blend)
    gate_text, gate_mask = gated_predictions(
        test_base_text,
        test_base_conf,
        test_blend_text,
        test_blend_conf,
        best_margin,
    )

    bicubic_cer, bicubic_exact, bicubic_cers, bicubic_exacts = scores(
        test_labels, test_base_text
    )
    blend_cer, blend_exact, blend_cers, blend_exacts = scores(
        test_labels, test_blend_text
    )
    gate_cer, gate_exact, gate_cers, gate_exacts = scores(test_labels, gate_text)

    results = pd.DataFrame({
        "index": np.arange(1, len(test_labels) + 1),
        "label": test_labels,
        "bicubic_text": test_base_text,
        "bicubic_confidence": test_base_conf,
        "bicubic_cer": bicubic_cers,
        "bicubic_exact": bicubic_exacts,
        "blend_text": test_blend_text,
        "blend_confidence": test_blend_conf,
        "blend_cer": blend_cers,
        "blend_exact": blend_exacts,
        "selected_method": np.where(gate_mask, "blend", "bicubic"),
        "gated_text": gate_text,
        "gated_cer": gate_cers,
        "gated_exact": gate_exacts,
    })
    gain = results.bicubic_cer - results.gated_cer
    summary = {
        "blend_alpha": alpha,
        "selected_confidence_margin": best_margin,
        "calibration_bicubic_cer": base_cer,
        "calibration_bicubic_exact": base_exact,
        "calibration_gated_cer": float(best["cer"]),
        "calibration_gated_exact": float(best["exact"]),
        "test_pairs": len(results),
        "bicubic_cer": bicubic_cer,
        "blend_cer": blend_cer,
        "gated_cer": gate_cer,
        "bicubic_exact": bicubic_exact,
        "blend_exact": blend_exact,
        "gated_exact": gate_exact,
        "selected_blend": int(gate_mask.sum()),
        "selected_bicubic": int((~gate_mask).sum()),
        "gated_improved": int((gain > 0).sum()),
        "gated_unchanged": int((gain == 0).sum()),
        "gated_worsened": int((gain < 0).sum()),
    }
    results_path = output / "confidence_gate_test_results.csv"
    summary_path = output / "confidence_gate_summary.json"
    results.to_csv(results_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    figure, left = plt.subplots(figsize=(8, 4.5))
    right = left.twinx()
    left.plot(calibration.margin, calibration.cer, "o-", color="tab:red", label="CER")
    right.plot(calibration.margin, calibration.exact, "s-", color="tab:blue", label="Exact")
    left.axvline(best_margin, color="black", linestyle="--", alpha=0.6, label=f"selected {best_margin:.3f}")
    right.axhline(base_exact, color="tab:blue", linestyle=":", alpha=0.5, label="bicubic exact")
    left.set(xlabel="Required blend confidence advantage", ylabel="PaddleOCR CER (lower is better)")
    right.set_ylabel("Exact accuracy (higher is better)")
    left.grid(alpha=0.25)
    left.legend(loc="upper left")
    right.legend(loc="upper right")
    plt.tight_layout()
    curve_path = output / "confidence_gate_calibration_curve.png"
    plt.savefig(curve_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    changed = results.index[gate_mask]
    if len(changed):
        changed_gain = gain.loc[changed]
        ranked = pd.concat([changed_gain.nsmallest(4), changed_gain.nlargest(4)]).index.unique().tolist()
    else:
        ranked = results.index[:8].tolist()
    figure, axes = plt.subplots(len(ranked), 4, figsize=(16, 2.2 * len(ranked)), squeeze=False)
    for row, result_index in enumerate(ranked):
        result = results.iloc[result_index]
        selected_image = test_blend[result_index] if gate_mask[result_index] else test_bicubic[result_index]
        pictures = [test_bicubic[result_index], test_blend[result_index], selected_image, test_hr[result_index]]
        names = ["bicubic", "blend", "gated", "hr"]
        texts = [result.bicubic_text, result.blend_text, result.gated_text, result.label]
        for column, (name, picture, prediction) in enumerate(zip(names, pictures, texts)):
            axes[row, column].imshow(picture)
            title = f"{name} | text: {prediction}"
            if column == 0:
                title += f"\nlabel: {result.label}"
            if column == 2:
                title += f"\nselected: {result.selected_method}"
            axes[row, column].set_title(title, fontsize=8)
            axes[row, column].axis("off")
    plt.tight_layout()
    visual_path = output / "confidence_gate_visual_comparison.png"
    plt.savefig(visual_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    print("saved:", calibration_path)
    print("saved:", curve_path)
    print("saved:", results_path)
    print("saved:", summary_path)
    print("saved:", visual_path)


if __name__ == "__main__":
    main()
