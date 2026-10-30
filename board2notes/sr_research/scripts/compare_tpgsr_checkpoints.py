#!/usr/bin/env python3
"""Compare bicubic, pixel/stroke TSRN-TL, and soft-prior TSRN-TL.

This reuses the completed soft-prior results and only runs the missing
pixel/stroke checkpoint through SR and PaddleOCR. No model is trained.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from train_tpgsr_textzoom import (
    TextZoomPairs,
    build_generator,
    build_recognizer,
    cer,
    choose_device,
    normalise_text,
    paddleocr_texts,
    project_paths,
    tensor_to_rgb,
    text_prior,
)


def main() -> None:
    _, source, textzoom, output = project_paths()
    device = choose_device()
    old_results_path = output / "tpgsr_results.csv"
    pixel_checkpoint = output / "tpgsr_single_stage_best.pt"
    soft_checkpoint = output / "tpgsr_soft_prior_best.pt"
    for required in (old_results_path, pixel_checkpoint, soft_checkpoint):
        if not required.exists():
            raise FileNotFoundError(required)

    previous = pd.read_csv(old_results_path).rename(columns={
        "tpgsr_psnr": "soft_prior_psnr",
        "tpgsr_ssim": "soft_prior_ssim",
        "tpgsr_text": "soft_prior_text",
        "tpgsr_cer": "soft_prior_cer",
        "tpgsr_exact": "soft_prior_exact",
    })
    dataset = TextZoomPairs(textzoom / "test" / "hard")
    if len(previous) != len(dataset):
        raise ValueError("Existing soft-prior results do not match TextZoom hard")

    recognizer = build_recognizer(source, device)
    pixel_generator = build_generator(source, device)
    saved = torch.load(pixel_checkpoint, map_location=device, weights_only=True)
    pixel_generator.load_state_dict(saved["generator"])
    pixel_generator.eval()
    print(f"device: {device}; pixel/stroke checkpoint epoch: {saved.get('epoch')}")

    pixel_images: list[np.ndarray] = []
    pixel_psnr: list[float] = []
    pixel_ssim: list[float] = []
    print(f"Restoring {len(dataset):,} TextZoom hard images with epoch-2 model...")
    with torch.no_grad():
        for index in range(len(dataset)):
            lr, hr, _, _ = dataset[index]
            lr_batch = lr.unsqueeze(0).to(device)
            prior = text_prior(recognizer, lr_batch)
            restored = ((pixel_generator(lr_batch, prior) + 1.0) / 2.0).clamp(0, 1)[0]
            restored_rgb = tensor_to_rgb(restored)
            hr_rgb = tensor_to_rgb(hr)
            pixel_images.append(restored_rgb)
            pixel_psnr.append(peak_signal_noise_ratio(hr_rgb, restored_rgb, data_range=255))
            pixel_ssim.append(structural_similarity(hr_rgb, restored_rgb, channel_axis=2, data_range=255))
            if (index + 1) % 200 == 0 or index + 1 == len(dataset):
                print(f"  SR {index + 1}/{len(dataset)}")

    print("OCR: pixel/stroke checkpoint")
    pixel_text = paddleocr_texts(pixel_images)
    results = previous.copy()
    results["pixel_stroke_psnr"] = pixel_psnr
    results["pixel_stroke_ssim"] = pixel_ssim
    results["pixel_stroke_text"] = pixel_text
    results["pixel_stroke_cer"] = [
        cer(label, prediction) for label, prediction in zip(results.label, pixel_text)
    ]
    results["pixel_stroke_exact"] = [
        int(normalise_text(label) == normalise_text(prediction))
        for label, prediction in zip(results.label, pixel_text)
    ]

    summary: dict[str, float | int] = {"pairs": len(results)}
    for name in ("bicubic", "pixel_stroke", "soft_prior", "hr"):
        for metric in ("psnr", "ssim", "cer", "exact"):
            column = f"{name}_{metric}"
            if column in results:
                summary[column] = float(results[column].mean())
        if name not in ("bicubic", "hr"):
            gain = results.bicubic_cer - results[f"{name}_cer"]
            summary[f"{name}_improved"] = int((gain > 0).sum())
            summary[f"{name}_unchanged"] = int((gain == 0).sum())
            summary[f"{name}_worsened"] = int((gain < 0).sum())

    result_path = output / "checkpoint_ablation_results.csv"
    summary_path = output / "checkpoint_ablation_summary.json"
    results.to_csv(result_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    difference = results.soft_prior_cer - results.pixel_stroke_cer
    ranked = pd.concat([difference.nsmallest(4), difference.nlargest(4)]).index.tolist()
    soft_generator = build_generator(source, device)
    soft_saved = torch.load(soft_checkpoint, map_location=device, weights_only=True)
    soft_generator.load_state_dict(soft_saved["generator"])
    soft_generator.eval()

    figure, axes = plt.subplots(len(ranked), 4, figsize=(16, 2.2 * len(ranked)))
    with torch.no_grad():
        for row, result_index in enumerate(ranked):
            lr, hr, label, _ = dataset[int(result_index)]
            lr_batch = lr.unsqueeze(0).to(device)
            prior = text_prior(recognizer, lr_batch)
            bicubic = F.interpolate(lr_batch, (32, 128), mode="bicubic", align_corners=False)[0]
            soft = ((soft_generator(lr_batch, prior) + 1.0) / 2.0).clamp(0, 1)[0]
            pictures = [tensor_to_rgb(bicubic), pixel_images[result_index], tensor_to_rgb(soft), tensor_to_rgb(hr)]
            names = ["bicubic", "pixel_stroke", "soft_prior", "hr"]
            result = results.iloc[result_index]
            for column, (name, picture) in enumerate(zip(names, pictures)):
                axes[row, column].imshow(picture)
                title = f"{name} | OCR: {result[f'{name}_text']}"
                if column == 0:
                    title += f"\nlabel: {label}"
                axes[row, column].set_title(title, fontsize=8)
                axes[row, column].axis("off")
    plt.tight_layout()
    figure_path = output / "checkpoint_ablation_visual.png"
    plt.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print("saved:", result_path)
    print("saved:", summary_path)
    print("saved:", figure_path)


if __name__ == "__main__":
    main()
