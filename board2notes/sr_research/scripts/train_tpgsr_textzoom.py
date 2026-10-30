#!/usr/bin/env python3
"""Train and evaluate a single-stage TPGSR reproduction on TextZoom.

This uses the official TPGSR TSRN_TL generator and the pretrained CRNN expected
by that repository.  It is intentionally described as a reproduction: the
authors did not publish a ready TPGSR generator checkpoint, and their complete
multi-stage training recipe is GPU-oriented.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/whiteboard-ocr-matplotlib")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import jiwer
import lmdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset


@dataclass
class Settings:
    epochs: int = 8
    batch_size: int = 16
    learning_rate: float = 1e-4
    prior_weight: float = 0.01
    gradient_weight: float = 0.20
    prior_warmup_epochs: int = 2
    prior_ramp_epochs: int = 2
    train_limit: int = 0
    val_limit: int = 512
    test_limit: int = 0
    seed: int = 20261018
    run_ocr: bool = True


def project_paths() -> tuple[Path, Path, Path, Path]:
    repo = Path(__file__).resolve().parents[1]
    assignment = repo.parent
    source = assignment / "models" / "TPGSR"
    textzoom = repo / "data" / "textzoom"
    output = repo / "outputs" / "tpgsr_textzoom"
    output.mkdir(parents=True, exist_ok=True)
    return repo, source, textzoom, output


class TextZoomPairs(Dataset):
    def __init__(self, folder: Path):
        self.folder = Path(folder)
        if not (self.folder / "data.mdb").exists():
            raise FileNotFoundError(f"TextZoom LMDB not found: {self.folder}")
        self.env = lmdb.open(
            str(self.folder), readonly=True, lock=False, readahead=False,
            meminit=False, max_readers=1,
        )
        with self.env.begin(write=False) as txn:
            self.count = int(txn.get(b"num-samples"))

    def __len__(self) -> int:
        return self.count

    @staticmethod
    def _decode(value: bytes) -> Image.Image:
        if value is None:
            raise KeyError("Image is missing from TextZoom LMDB")
        return Image.open(BytesIO(value)).convert("RGB")

    @staticmethod
    def _tensor(image: Image.Image, size: tuple[int, int]) -> torch.Tensor:
        array = np.asarray(image.resize(size, Image.Resampling.BICUBIC), dtype=np.float32).copy() / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    def __getitem__(self, zero_index: int):
        index = zero_index + 1
        suffix = f"{index:09d}".encode()
        with self.env.begin(write=False) as txn:
            label_value = txn.get(b"label-" + suffix)
            lr_value = txn.get(b"image_lr-" + suffix)
            hr_value = txn.get(b"image_hr-" + suffix)
        label = label_value.decode("utf-8") if label_value else ""
        lr = self._tensor(self._decode(lr_value), (64, 16))
        hr = self._tensor(self._decode(hr_value), (128, 32))
        return lr, hr, label, index


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_recognizer(source: Path, device: torch.device) -> nn.Module:
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from model.crnn.model import Model

    options = SimpleNamespace(
        Transformation="None", FeatureExtraction="ResNet",
        SequenceModeling="None", Prediction="CTC", num_fiducial=20,
        input_channel=1, output_channel=512, hidden_size=256,
        character="-0123456789abcdefghijklmnopqrstuvwxyz", num_class=37,
    )
    recognizer = Model(options)
    weight_path = source / "None-ResNet-None-CTC.pth"
    if not weight_path.exists():
        raise FileNotFoundError(
            f"Missing pretrained recognizer: {weight_path}\n"
            "Run the download/setup cell in the notebook first."
        )
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    state = {key.replace("module.", ""): value for key, value in state.items()}
    recognizer.load_state_dict(state, strict=True)
    recognizer.to(device).eval()
    for parameter in recognizer.parameters():
        parameter.requires_grad_(False)
    return recognizer


def build_generator(source: Path, device: torch.device) -> nn.Module:
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from model.tsrn import TSRN_TL

    return TSRN_TL(
        scale_factor=2, width=128, height=32, STN=False,
        srb_nums=5, mask=False, hidden_units=32,
        text_emb=37, out_text_channels=32,
    ).to(device)


def recognizer_input(images: torch.Tensor) -> torch.Tensor:
    images = F.interpolate(images, (32, 100), mode="bicubic", align_corners=False)
    return 0.299 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]


def text_prior(recognizer: nn.Module, lr: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        logits = recognizer(recognizer_input(lr))  # time, batch, classes
        probabilities = torch.softmax(logits, dim=-1)
    return probabilities.permute(1, 2, 0).unsqueeze(2)  # batch, classes, 1, time


def gradient_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_x = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
    true_x = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_y = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
    true_y = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_x, true_x) + F.l1_loss(pred_y, true_y)


def soft_prior_loss(
    recognizer: nn.Module, prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Match soft CRNN character probabilities instead of forcing a hard CTC label."""
    prediction_logits = recognizer(recognizer_input(prediction))
    with torch.no_grad():
        target_logits = recognizer(recognizer_input(target))
        target_probabilities = torch.softmax(target_logits, dim=-1)
    prediction_probabilities = torch.softmax(prediction_logits, dim=-1)
    probability_l1 = F.l1_loss(prediction_probabilities, target_probabilities)
    # batchmean sums over time and classes, so divide by time for a stable scale.
    probability_kl = F.kl_div(
        F.log_softmax(prediction_logits, dim=-1),
        target_probabilities,
        reduction="batchmean",
    ) / prediction_logits.shape[0]
    return probability_l1 + probability_kl


def losses(generator, recognizer, lr, hr, settings, prior_weight=None):
    prior = text_prior(recognizer, lr)
    restored = (generator(lr, prior) + 1.0) / 2.0
    restored = restored.clamp(0.0, 1.0)
    pixel = F.l1_loss(restored, hr)
    edge = gradient_loss(restored, hr)

    weight = settings.prior_weight if prior_weight is None else prior_weight
    if weight > 0:
        prior_matching = soft_prior_loss(recognizer, restored, hr)
    else:
        prior_matching = restored.new_zeros(())
    total = pixel + settings.gradient_weight * edge + weight * prior_matching
    return total, pixel, edge, prior_matching, restored


def limited_subset(dataset, indices: np.ndarray, limit: int) -> Subset:
    if limit and limit < len(indices):
        indices = indices[:limit]
    return Subset(dataset, indices.tolist())


def train(
    settings: Settings, source: Path, textzoom: Path, output: Path,
    device: torch.device, resume: bool = False,
):
    full_train = ConcatDataset([
        TextZoomPairs(textzoom / "train1"),
        TextZoomPairs(textzoom / "train2"),
    ])
    rng = np.random.default_rng(settings.seed)
    order = rng.permutation(len(full_train))
    val_count = min(1000, max(1, len(order) // 10))
    val_indices = order[:val_count]
    train_indices = order[val_count:]
    train_set = limited_subset(full_train, train_indices, settings.train_limit)
    val_set = limited_subset(full_train, val_indices, settings.val_limit)
    train_loader = DataLoader(train_set, batch_size=settings.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=settings.batch_size, shuffle=False, num_workers=0)

    recognizer = build_recognizer(source, device)
    generator = build_generator(source, device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=settings.learning_rate)
    checkpoint_path = output / "tpgsr_soft_prior_best.pt"
    start_epoch = 1
    if resume:
        resume_path = output / "tpgsr_single_stage_best.pt"
        if not resume_path.exists():
            raise FileNotFoundError(f"Warm-up checkpoint is missing: {resume_path}")
        saved = torch.load(resume_path, map_location=device, weights_only=True)
        generator.load_state_dict(saved["generator"])
        start_epoch = int(saved.get("epoch", settings.prior_warmup_epochs)) + 1
        print(f"resumed generator weights from epoch {start_epoch - 1}: {resume_path}")
    best_val = float("inf")
    history = []

    print(f"device: {device}; train={len(train_set):,}; val={len(val_set):,}")
    print("official generator: TSRN_TL; prior: frozen CRNN probabilities")
    print(
        f"loss schedule: {settings.prior_warmup_epochs} pixel/stroke warm-up epochs, "
        f"then soft CRNN-prior matching ramps to weight {settings.prior_weight}"
    )
    for epoch in range(start_epoch, settings.epochs + 1):
        if epoch <= settings.prior_warmup_epochs:
            active_prior_weight = 0.0
        else:
            ramp_step = epoch - settings.prior_warmup_epochs
            ramp = min(1.0, ramp_step / max(1, settings.prior_ramp_epochs))
            active_prior_weight = settings.prior_weight * ramp
        generator.train()
        totals = []
        for lr, hr, _, _ in train_loader:
            lr, hr = lr.to(device), hr.to(device)
            optimizer.zero_grad(set_to_none=True)
            total, pixel, edge, prior_matching, _ = losses(
                generator, recognizer, lr, hr, settings,
                prior_weight=active_prior_weight,
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 5.0)
            optimizer.step()
            totals.append([total.item(), pixel.item(), edge.item(), prior_matching.item()])

        generator.eval()
        validation = []
        with torch.no_grad():
            for lr, hr, _, _ in val_loader:
                lr, hr = lr.to(device), hr.to(device)
                # Always use the final soft-prior weight for a comparable
                # validation score across warm-up and guided epochs.
                values = losses(
                    generator, recognizer, lr, hr, settings,
                    prior_weight=settings.prior_weight,
                )
                validation.append([value.item() for value in values[:4]])
        train_mean = np.mean(totals, axis=0)
        val_mean = np.mean(validation, axis=0)
        improved = val_mean[0] < best_val
        if improved:
            best_val = float(val_mean[0])
            torch.save(
                {"generator": generator.state_dict(), "epoch": epoch,
                 "settings": asdict(settings), "val_loss": best_val},
                checkpoint_path,
            )
        history.append({
            "epoch": epoch, "train_total": train_mean[0], "train_l1": train_mean[1],
            "train_gradient": train_mean[2], "train_soft_prior": train_mean[3],
            "val_total": val_mean[0], "val_l1": val_mean[1],
            "val_gradient": val_mean[2], "val_soft_prior": val_mean[3],
        })
        marker = " * best" if improved else ""
        print(
            f"epoch {epoch:02d}: train={train_mean[0]:.4f} "
            f"val={val_mean[0]:.4f} val_L1={val_mean[1]:.4f} "
            f"val_soft_prior={val_mean[3]:.4f} "
            f"train_prior_weight={active_prior_weight:.4f}{marker}"
        )

    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    print("checkpoint:", checkpoint_path)
    return checkpoint_path


def normalise_text(text: str) -> str:
    return re.sub(r"[^0-9a-z]", "", str(text).lower())


def cer(reference: str, prediction: str) -> float:
    return min(float(jiwer.cer(normalise_text(reference), normalise_text(prediction) or " ")), 1.0)


def tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    return (tensor.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255).round().astype(np.uint8)


def paddleocr_texts(images: list[np.ndarray], chunk_size: int = 128) -> list[str]:
    from paddleocr import TextRecognition

    recognizer = TextRecognition(model_name="PP-OCRv6_medium_rec")
    texts: list[str] = []
    for start in range(0, len(images), chunk_size):
        batch = [cv2.cvtColor(image, cv2.COLOR_RGB2BGR) for image in images[start:start + chunk_size]]
        predictions = recognizer.predict(batch)
        texts.extend(str(prediction["rec_text"]) for prediction in predictions)
        print(f"  PaddleOCR {min(start + chunk_size, len(images))}/{len(images)}")
    return texts


def evaluate(settings: Settings, checkpoint: Path, source: Path, textzoom: Path,
             output: Path, device: torch.device):
    dataset = TextZoomPairs(textzoom / "test" / "hard")
    indices = np.arange(len(dataset))
    if settings.test_limit and settings.test_limit < len(indices):
        indices = indices[:settings.test_limit]

    recognizer = build_recognizer(source, device)
    generator = build_generator(source, device)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    generator.load_state_dict(saved["generator"])
    generator.eval()

    rows, images = [], {"bicubic": [], "tpgsr": [], "hr": []}
    print(f"Evaluating {len(indices):,} untouched TextZoom hard pairs...")
    with torch.no_grad():
        for position, index in enumerate(indices, start=1):
            lr, hr, label, sample_index = dataset[int(index)]
            lr_batch = lr.unsqueeze(0).to(device)
            prior = text_prior(recognizer, lr_batch)
            restored = ((generator(lr_batch, prior) + 1.0) / 2.0).clamp(0, 1)[0]
            bicubic = F.interpolate(lr_batch, (32, 128), mode="bicubic", align_corners=False).clamp(0, 1)[0]
            rgb = {"bicubic": tensor_to_rgb(bicubic), "tpgsr": tensor_to_rgb(restored), "hr": tensor_to_rgb(hr)}
            for name in images:
                images[name].append(rgb[name])
            row = {"index": sample_index, "label": label}
            for name in images:
                row[f"{name}_psnr"] = peak_signal_noise_ratio(rgb["hr"], rgb[name], data_range=255)
                row[f"{name}_ssim"] = structural_similarity(rgb["hr"], rgb[name], channel_axis=2, data_range=255)
            rows.append(row)
            if position % 200 == 0 or position == len(indices):
                print(f"  SR {position}/{len(indices)}")

    results = pd.DataFrame(rows)
    if settings.run_ocr:
        for name in images:
            print(f"OCR: {name}")
            texts = paddleocr_texts(images[name])
            results[f"{name}_text"] = texts
            results[f"{name}_cer"] = [cer(label, text) for label, text in zip(results.label, texts)]
            results[f"{name}_exact"] = [
                int(normalise_text(label) == normalise_text(text))
                for label, text in zip(results.label, texts)
            ]

    result_path = output / "tpgsr_results.csv"
    results.to_csv(result_path, index=False)
    summary = {
        "pairs": len(results),
        "bicubic_psnr": float(results.bicubic_psnr.mean()),
        "tpgsr_psnr": float(results.tpgsr_psnr.mean()),
        "bicubic_ssim": float(results.bicubic_ssim.mean()),
        "tpgsr_ssim": float(results.tpgsr_ssim.mean()),
    }
    if settings.run_ocr:
        gain = results.bicubic_cer - results.tpgsr_cer
        summary.update({
            "bicubic_cer": float(results.bicubic_cer.mean()),
            "tpgsr_cer": float(results.tpgsr_cer.mean()),
            "bicubic_exact": float(results.bicubic_exact.mean()),
            "tpgsr_exact": float(results.tpgsr_exact.mean()),
            "improved": int((gain > 0).sum()),
            "unchanged": int((gain == 0).sum()),
            "worsened": int((gain < 0).sum()),
        })
        ranked = pd.concat([gain.nsmallest(4), gain.nlargest(4)]).index.tolist()
    else:
        gain = results.tpgsr_psnr - results.bicubic_psnr
        ranked = pd.concat([gain.nsmallest(4), gain.nlargest(4)]).index.tolist()

    with (output / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2))

    figure, axes = plt.subplots(len(ranked), 3, figsize=(13, 2.2 * len(ranked)))
    for row_number, result_index in enumerate(ranked):
        result = results.iloc[result_index]
        for column, name in enumerate(["bicubic", "tpgsr", "hr"]):
            axes[row_number, column].imshow(images[name][result_index])
            title = name
            if settings.run_ocr:
                title += f" | OCR: {result[f'{name}_text']}"
            if column == 0:
                title += f"\nlabel: {result.label}"
            axes[row_number, column].set_title(title, fontsize=8)
            axes[row_number, column].axis("off")
    plt.tight_layout()
    figure_path = output / "tpgsr_visual_comparison.png"
    plt.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print("saved:", result_path)
    print("saved:", figure_path)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=512)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    settings = Settings(
        epochs=args.epochs, batch_size=args.batch_size,
        train_limit=args.train_limit, val_limit=args.val_limit,
        test_limit=args.test_limit, run_ocr=not args.no_ocr,
    )
    seed_everything(settings.seed)
    _, source, textzoom, output = project_paths()
    if not source.is_dir():
        raise FileNotFoundError(f"Official TPGSR source is missing: {source}")
    device = choose_device()
    checkpoint = output / "tpgsr_soft_prior_best.pt"
    if not args.evaluate_only:
        checkpoint = train(settings, source, textzoom, output, device, resume=args.resume)
    if not checkpoint.exists():
        raise FileNotFoundError(f"No trained checkpoint: {checkpoint}")
    evaluate(settings, checkpoint, source, textzoom, output, device)


if __name__ == "__main__":
    main()
