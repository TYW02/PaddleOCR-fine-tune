"""OCR-safe, label-preserving augmentation for IAM handwritten line crops.

The clean dataset remains unchanged. This module is intended for previewing
and for adapting into a training data loader. Validation and test data must not
use these random transformations.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image, ImageEnhance


def load_policy(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _as_rgb(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"))
    else:
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    return arr.astype(np.uint8, copy=False)


def _weighted_choice(items: dict[str, Any], rng: np.random.Generator) -> str:
    names = list(items)
    weights = np.asarray([items[name].get("weight", 1.0) for name in names], dtype=float)
    weights /= weights.sum()
    return str(rng.choice(names, p=weights))


def rotate(image: np.ndarray, cfg: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    h, w = image.shape[:2]
    # A fixed angle that is harmless for a square image can clip a very wide
    # text line. Limit the angle by the permitted canvas-height growth, then
    # expand the canvas so no characters are removed.
    growth = float(cfg.get("max_height_growth_fraction", 0.18))
    aspect_safe_degrees = float(np.degrees(np.arctan2(growth * h, max(w, 1))))
    max_degrees = min(float(cfg["max_degrees"]), aspect_safe_degrees)
    angle = float(rng.uniform(-max_degrees, max_degrees))
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    new_w = int(np.ceil(h * sine + w * cosine))
    new_h = int(np.ceil(h * cosine + w * sine))
    matrix[0, 2] += new_w / 2 - w / 2
    matrix[1, 2] += new_h / 2 - h / 2
    return cv2.warpAffine(
        image,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def perspective(image: np.ndarray, cfg: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    h, w = image.shape[:2]
    source = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    max_x = max(1.0, w * float(cfg["max_x_fraction"]))
    max_y = max(1.0, h * float(cfg["max_y_fraction"]))
    offsets = np.column_stack(
        [rng.uniform(-max_x, max_x, 4), rng.uniform(-max_y, max_y, 4)]
    ).astype(np.float32)
    target = source + offsets
    target[:, 0] = np.clip(target[:, 0], 0, w - 1)
    target[:, 1] = np.clip(target[:, 1], 0, h - 1)
    matrix = cv2.getPerspectiveTransform(source, target)
    return cv2.warpPerspective(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def add_margin(image: np.ndarray, cfg: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    h, w = image.shape[:2]
    pad_x = int(rng.uniform(0.005, cfg["max_x_fraction"]) * w)
    pad_y = int(rng.uniform(0.01, cfg["max_y_fraction"]) * h)
    return cv2.copyMakeBorder(
        image,
        pad_y,
        pad_y,
        pad_x,
        pad_x,
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )


def brightness_contrast(
    image: np.ndarray, cfg: dict[str, Any], rng: np.random.Generator
) -> np.ndarray:
    pil = Image.fromarray(image)
    brightness = float(rng.uniform(*cfg["brightness_range"]))
    contrast = float(rng.uniform(*cfg["contrast_range"]))
    pil = ImageEnhance.Brightness(pil).enhance(brightness)
    pil = ImageEnhance.Contrast(pil).enhance(contrast)
    return np.asarray(pil, dtype=np.uint8)


def uneven_illumination(
    image: np.ndarray, cfg: dict[str, Any], rng: np.random.Generator
) -> np.ndarray:
    h, w = image.shape[:2]
    start = float(rng.uniform(cfg["min_gain"], 1.0))
    end = float(rng.uniform(1.0, cfg["max_gain"]))
    if rng.random() < 0.5:
        start, end = end, start
    horizontal = np.linspace(start, end, w, dtype=np.float32)
    vertical_shift = np.linspace(0.96, 1.04, h, dtype=np.float32)[:, None]
    field = np.clip(vertical_shift * horizontal[None, :], cfg["min_gain"], cfg["max_gain"])
    out = image.astype(np.float32) * field[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def gaussian_blur(
    image: np.ndarray, cfg: dict[str, Any], rng: np.random.Generator
) -> np.ndarray:
    sigma = float(rng.uniform(*cfg["sigma_range"]))
    return cv2.GaussianBlur(image, (3, 3), sigmaX=sigma)


def motion_blur(image: np.ndarray, cfg: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    size = int(rng.choice(cfg["kernel_choices"]))
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0
    angle = float(rng.uniform(-20, 20))
    matrix = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (size, size))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(image, -1, kernel)


def resolution_reduction(
    image: np.ndarray, cfg: dict[str, Any], rng: np.random.Generator
) -> np.ndarray:
    h, w = image.shape[:2]
    scale = float(rng.uniform(*cfg["scale_range"]))
    small = cv2.resize(
        image,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def gaussian_noise(
    image: np.ndarray, cfg: dict[str, Any], rng: np.random.Generator
) -> np.ndarray:
    sigma = float(rng.uniform(*cfg["sigma_range"]))
    noise = rng.normal(0.0, sigma, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def jpeg_compression(
    image: np.ndarray, cfg: dict[str, Any], rng: np.random.Generator
) -> np.ndarray:
    quality = int(rng.integers(cfg["quality_range"][0], cfg["quality_range"][1] + 1))
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=quality, optimize=False)
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("RGB"), dtype=np.uint8)


TRANSFORMS = {
    "rotation": rotate,
    "perspective": perspective,
    "add_margin": add_margin,
    "brightness_contrast": brightness_contrast,
    "uneven_illumination": uneven_illumination,
    "gaussian_blur": gaussian_blur,
    "motion_blur": motion_blur,
    "resolution_reduction": resolution_reduction,
    "gaussian_noise": gaussian_noise,
    "jpeg_compression": jpeg_compression,
}


def apply_named(
    image: Image.Image | np.ndarray,
    name: str,
    policy: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    for group in policy["groups"].values():
        if name in group["transforms"]:
            return TRANSFORMS[name](_as_rgb(image), group["transforms"][name], rng)
    raise KeyError(f"Unknown transform: {name}")


def augment(
    image: Image.Image | np.ndarray,
    policy: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[str]]:
    output = _as_rgb(image)
    applied: list[str] = []
    for group in policy["groups"].values():
        if rng.random() >= float(group["probability"]):
            continue
        name = _weighted_choice(group["transforms"], rng)
        output = TRANSFORMS[name](output, group["transforms"][name], rng)
        applied.append(name)
    return output, applied
