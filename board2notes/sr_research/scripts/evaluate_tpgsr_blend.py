#!/usr/bin/env python3
"""Tune bicubic/TPGSR blending on calibration data, then test once."""

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
import torch.nn.functional as F
from paddleocr import TextRecognition
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import ConcatDataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_tpgsr_textzoom import (
    Settings,
    TextZoomPairs,
    build_generator,
    build_recognizer,
    cer,
    choose_device,
    normalise_text,
    project_paths,
    tensor_to_rgb,
    text_prior,
)


ALPHAS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0]


def recognise(model: TextRecognition, images: list[np.ndarray], chunk_size: int = 128) -> list[str]:
    texts: list[str] = []
    for start in range(0, len(images), chunk_size):
        batch = [cv2.cvtColor(image, cv2.COLOR_RGB2BGR) for image in images[start:start + chunk_size]]
        predictions = model.predict(batch)
        texts.extend(str(prediction["rec_text"]) for prediction in predictions)
        print(f"  PaddleOCR {min(start + chunk_size, len(images))}/{len(images)}")
    return texts


def blend(bicubic: list[np.ndarray], neural: list[np.ndarray], alpha: float) -> list[np.ndarray]:
    return [
        np.clip((1.0 - alpha) * base.astype(np.float32) + alpha * sr.astype(np.float32), 0, 255)
        .round().astype(np.uint8)
        for base, sr in zip(bicubic, neural)
    ]


def generate_images(dataset, indices, generator, recognizer, device):
    labels, bicubic_images, neural_images, hr_images = [], [], [], []
    with torch.no_grad():
        for position, index in enumerate(indices, start=1):
            lr, hr, label, _ = dataset[int(index)]
            lr_batch = lr.unsqueeze(0).to(device)
            prior = text_prior(recognizer, lr_batch)
            neural = ((generator(lr_batch, prior) + 1.0) / 2.0).clamp(0, 1)[0]
            bicubic = F.interpolate(lr_batch, (32, 128), mode="bicubic", align_corners=False)[0]
            labels.append(label)
            bicubic_images.append(tensor_to_rgb(bicubic))
            neural_images.append(tensor_to_rgb(neural))
            hr_images.append(tensor_to_rgb(hr))
            if position % 200 == 0 or position == len(indices):
                print(f"  SR {position}/{len(indices)}")
    return labels, bicubic_images, neural_images, hr_images


def scores(labels: list[str], predictions: list[str]) -> tuple[float, float, list[float], list[int]]:
    cers = [cer(label, prediction) for label, prediction in zip(labels, predictions)]
    exacts = [
        int(normalise_text(label) == normalise_text(prediction))
        for label, prediction in zip(labels, predictions)
    ]
    return float(np.mean(cers)), float(np.mean(exacts)), cers, exacts


def main() -> None:
    _, source, textzoom, output = project_paths()
    device = choose_device()
    checkpoint = output / "tpgsr_soft_prior_best.pt"
    previous_path = output / "tpgsr_results.csv"
    if not checkpoint.exists() or not previous_path.exists():
        raise FileNotFoundError("Run the completed soft-prior experiment first")

    recognizer = build_recognizer(source, device)
    generator = build_generator(source, device)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    generator.load_state_dict(saved["generator"])
    generator.eval()
    paddle = TextRecognition(model_name="PP-OCRv6_medium_rec")
    print(f"device: {device}; soft-prior checkpoint epoch: {saved.get('epoch')}")

    # The first 512 held-out pairs were used for validation. The remaining 488
    # of the original 1,000 held-out pairs were not used for training or model selection.
    full_train = ConcatDataset([
        TextZoomPairs(textzoom / "train1"),
        TextZoomPairs(textzoom / "train2"),
    ])
    rng = np.random.default_rng(Settings().seed)
    order = rng.permutation(len(full_train))
    calibration_indices = order[512:1000]
    print(f"Generating {len(calibration_indices)} untouched calibration pairs...")
    labels, bicubic_images, neural_images, _ = generate_images(
        full_train, calibration_indices, generator, recognizer, device
    )

    calibration_rows = []
    for alpha in ALPHAS:
        print(f"Calibration OCR: alpha={alpha:.2f}")
        predictions = recognise(paddle, blend(bicubic_images, neural_images, alpha))
        mean_cer, exact, _, _ = scores(labels, predictions)
        calibration_rows.append({"alpha": alpha, "cer": mean_cer, "exact": exact})
        print(f"  CER={mean_cer:.4f}; exact={exact:.4f}")
    calibration = pd.DataFrame(calibration_rows)
    best = min(calibration_rows, key=lambda row: (row["cer"], -row["exact"], row["alpha"]))
    best_alpha = float(best["alpha"])
    calibration_path = output / "blend_calibration.csv"
    calibration.to_csv(calibration_path, index=False)
    print(f"Selected alpha={best_alpha:.2f} using calibration CER only")

    previous = pd.read_csv(previous_path)
    hard = TextZoomPairs(textzoom / "test" / "hard")
    test_indices = np.arange(len(hard))
    print(f"Generating {len(test_indices):,} untouched hard-test pairs...")
    test_labels, test_bicubic, test_neural, test_hr = generate_images(
        hard, test_indices, generator, recognizer, device
    )
    test_blend = blend(test_bicubic, test_neural, best_alpha)
    if best_alpha == 0.0:
        blend_text = previous.bicubic_text.fillna("").astype(str).tolist()
    elif best_alpha == 1.0:
        blend_text = previous.tpgsr_text.fillna("").astype(str).tolist()
    else:
        print(f"Test OCR: selected alpha={best_alpha:.2f}")
        blend_text = recognise(paddle, test_blend)
    blend_cer, blend_exact, blend_cers, blend_exacts = scores(test_labels, blend_text)

    results = previous.copy()
    results["blend_alpha"] = best_alpha
    results["blend_text"] = blend_text
    results["blend_cer"] = blend_cers
    results["blend_exact"] = blend_exacts
    results["blend_psnr"] = [
        peak_signal_noise_ratio(hr, prediction, data_range=255)
        for hr, prediction in zip(test_hr, test_blend)
    ]
    results["blend_ssim"] = [
        structural_similarity(hr, prediction, channel_axis=2, data_range=255)
        for hr, prediction in zip(test_hr, test_blend)
    ]
    gain = results.bicubic_cer - results.blend_cer
    summary = {
        "selected_alpha": best_alpha,
        "calibration_cer": float(best["cer"]),
        "calibration_exact": float(best["exact"]),
        "test_pairs": len(results),
        "bicubic_cer": float(results.bicubic_cer.mean()),
        "blend_cer": blend_cer,
        "tpgsr_cer": float(results.tpgsr_cer.mean()),
        "bicubic_exact": float(results.bicubic_exact.mean()),
        "blend_exact": blend_exact,
        "tpgsr_exact": float(results.tpgsr_exact.mean()),
        "bicubic_psnr": float(results.bicubic_psnr.mean()),
        "blend_psnr": float(results.blend_psnr.mean()),
        "tpgsr_psnr": float(results.tpgsr_psnr.mean()),
        "bicubic_ssim": float(results.bicubic_ssim.mean()),
        "blend_ssim": float(results.blend_ssim.mean()),
        "tpgsr_ssim": float(results.tpgsr_ssim.mean()),
        "blend_improved": int((gain > 0).sum()),
        "blend_unchanged": int((gain == 0).sum()),
        "blend_worsened": int((gain < 0).sum()),
    }
    results_path = output / "blend_test_results.csv"
    summary_path = output / "blend_summary.json"
    results.to_csv(results_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    figure, left = plt.subplots(figsize=(8, 4.5))
    right = left.twinx()
    left.plot(calibration.alpha, calibration.cer, "o-", color="tab:red", label="CER")
    right.plot(calibration.alpha, calibration.exact, "s-", color="tab:blue", label="Exact")
    left.axvline(best_alpha, color="black", linestyle="--", alpha=0.6, label=f"selected {best_alpha:.2f}")
    left.set(xlabel="TPGSR blend strength (alpha)", ylabel="PaddleOCR CER (lower is better)")
    right.set_ylabel("Exact accuracy (higher is better)")
    left.grid(alpha=0.25)
    left.legend(loc="upper left")
    right.legend(loc="upper right")
    plt.tight_layout()
    curve_path = output / "blend_calibration_curve.png"
    plt.savefig(curve_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    ranked = pd.concat([gain.nsmallest(4), gain.nlargest(4)]).index.tolist()
    figure, axes = plt.subplots(len(ranked), 4, figsize=(16, 2.2 * len(ranked)))
    for row, result_index in enumerate(ranked):
        result = results.iloc[result_index]
        pictures = [test_bicubic[result_index], test_neural[result_index], test_blend[result_index], test_hr[result_index]]
        names = ["bicubic", "tpgsr", "blend", "hr"]
        for column, (name, picture) in enumerate(zip(names, pictures)):
            axes[row, column].imshow(picture)
            title = f"{name} | OCR: {result[f'{name}_text']}"
            if column == 0:
                title += f"\nlabel: {result.label}"
            axes[row, column].set_title(title, fontsize=8)
            axes[row, column].axis("off")
    plt.tight_layout()
    visual_path = output / "blend_visual_comparison.png"
    plt.savefig(visual_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print("saved:", calibration_path)
    print("saved:", curve_path)
    print("saved:", results_path)
    print("saved:", summary_path)
    print("saved:", visual_path)


if __name__ == "__main__":
    main()
