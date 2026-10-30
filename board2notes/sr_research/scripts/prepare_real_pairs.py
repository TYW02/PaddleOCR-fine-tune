#!/usr/bin/env python3
"""Prepare reviewable real and synthetic whiteboard line pairs.

The script never modifies source photographs. It:
1. reads the curated photo-level pairing plan;
2. detects line candidates on each sharp close photograph with PaddleOCR;
3. aligns the close photograph to its medium/far/side photograph with SIFT +
   a robust homography;
4. warps the same physical line from both photographs into one coordinate
   system; and
5. writes lossless PNG crops, draft transcriptions, diagnostics and manifests.

OCR transcriptions are suggestions only. A human must verify them before the
manifest is used for recogniser evaluation or supervised training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()


@dataclass
class Detection:
    polygon: np.ndarray
    text: str
    confidence: float


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()
    project = here.parents[1]
    assignment = project.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--curated-root",
        type=Path,
        default=assignment / "17 Oct_filtered",
        help="Folder containing pairing_plan.csv and 17oct_board01/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project / "outputs" / "paired_dataset_17oct_pilot",
    )
    parser.add_argument(
        "--max-inputs",
        type=int,
        default=2,
        help="Number of input photographs to process. Use 0 for every row.",
    )
    parser.add_argument(
        "--roles",
        default="medium_input,far_input",
        help="Comma-separated input roles. Add side_angle_input after reviewing the pilot.",
    )
    parser.add_argument("--synthetic-per-target", type=int, default=2)
    parser.add_argument(
        "--target-height",
        type=int,
        default=0,
        help=(
            "Saved crop height. The default 0 preserves native close-photo detail; "
            "resize only inside the training data loader."
        ),
    )
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--force", action="store_true", help="Replace existing output files.")
    return parser.parse_args()


def read_heic_bgr(path: Path) -> np.ndarray:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def locate_photo(board_root: Path, file_name: str) -> Path:
    matches = list(board_root.glob(f"*/{file_name}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one selected copy of {file_name}; found {len(matches)}")
    return matches[0]


def resize_for_work(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    scale = min(1.0, max_side / max(image.shape[:2]))
    if scale == 1.0:
        return image, scale
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), scale


def estimate_homography(target: np.ndarray, source: np.ndarray) -> tuple[np.ndarray | None, dict]:
    """Return the full-resolution transform target -> source."""
    target_small, st = resize_for_work(target, 1800)
    source_small, ss = resize_for_work(source, 1800)
    sift = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.015, edgeThreshold=20)
    kt, dt = sift.detectAndCompute(cv2.cvtColor(target_small, cv2.COLOR_BGR2GRAY), None)
    ks, ds = sift.detectAndCompute(cv2.cvtColor(source_small, cv2.COLOR_BGR2GRAY), None)
    stats = {"target_keypoints": len(kt), "source_keypoints": len(ks), "matches": 0,
             "inliers": 0, "inlier_ratio": 0.0, "median_error_px": math.inf}
    if dt is None or ds is None:
        return None, stats

    matches = cv2.BFMatcher().knnMatch(dt, ds, k=2)
    good = [a for a, b in matches if a.distance < 0.72 * b.distance]
    stats["matches"] = len(good)
    if len(good) < 12:
        return None, stats

    pt = np.float32([kt[m.queryIdx].pt for m in good])
    ps = np.float32([ks[m.trainIdx].pt for m in good])
    h_small, mask = cv2.findHomography(pt, ps, cv2.USAC_MAGSAC, 3.0,
                                       maxIters=10000, confidence=0.999)
    if h_small is None or mask is None:
        return None, stats

    mask = mask.ravel().astype(bool)
    predicted = cv2.perspectiveTransform(pt.reshape(-1, 1, 2), h_small).reshape(-1, 2)
    errors = np.linalg.norm(predicted - ps, axis=1)
    stats.update(inliers=int(mask.sum()), inlier_ratio=float(mask.mean()),
                 median_error_px=float(np.median(errors[mask])))

    # H_full = source_scale^-1 * H_small * target_scale
    target_to_small = np.diag([st, st, 1.0])
    source_to_small = np.diag([ss, ss, 1.0])
    h_full = np.linalg.inv(source_to_small) @ h_small @ target_to_small
    h_full /= h_full[2, 2]
    if stats["inliers"] < 20 or stats["inlier_ratio"] < 0.25:
        return None, stats
    if not np.isfinite(h_full).all() or np.linalg.matrix_rank(h_full) < 3:
        return None, stats
    try:
        np.linalg.inv(h_full)
    except np.linalg.LinAlgError:
        return None, stats
    return h_full, stats


def paddle_detector():
    from paddleocr import PaddleOCR

    return PaddleOCR(
        text_detection_model_name="PP-OCRv6_medium_det",
        text_recognition_model_name="PP-OCRv6_medium_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def extract_result_dict(result) -> dict:
    data = result.json if hasattr(result, "json") else result
    data = data() if callable(data) else data
    return data.get("res", data)


def detect_lines(ocr, image: np.ndarray) -> list[Detection]:
    work, scale = resize_for_work(image, 2200)
    results = list(ocr.predict(
        work,
        text_det_thresh=0.20,
        text_det_box_thresh=0.40,
        text_det_unclip_ratio=2.00,
        text_rec_score_thresh=0.00,
    ))
    if not results:
        return []
    data = extract_result_dict(results[0])
    polygons = data.get("rec_polys") or data.get("dt_polys") or []
    texts = data.get("rec_texts") or [""] * len(polygons)
    scores = data.get("rec_scores") or [0.0] * len(polygons)
    detections = []
    for polygon, text, score in zip(polygons, texts, scores):
        p = np.asarray(polygon, np.float32) / scale
        width = 0.5 * (np.linalg.norm(p[1] - p[0]) + np.linalg.norm(p[2] - p[3]))
        height = 0.5 * (np.linalg.norm(p[3] - p[0]) + np.linalg.norm(p[2] - p[1]))
        if width < 35 or height < 12 or width * height < 1000:
            continue
        detections.append(Detection(polygon=order_quad(p), text=str(text), confidence=float(score)))
    return merge_line_fragments(detections)


def order_quad(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, np.float32).reshape(4, 2)
    s = points.sum(axis=1)
    d = np.diff(points, axis=1).ravel()
    return np.asarray([points[np.argmin(s)], points[np.argmin(d)],
                       points[np.argmax(s)], points[np.argmax(d)]], np.float32)


def bounds(det: Detection) -> tuple[float, float, float, float]:
    p = det.polygon
    return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())


def merge_line_fragments(detections: list[Detection]) -> list[Detection]:
    """Join PaddleOCR fragments that occupy the same handwritten row."""
    pending = sorted(detections, key=lambda d: ((bounds(d)[1] + bounds(d)[3]) / 2, bounds(d)[0]))
    groups: list[list[Detection]] = []
    for det in pending:
        x0, y0, x1, y1 = bounds(det)
        cy, height = (y0 + y1) / 2, y1 - y0
        best = None
        for group in groups:
            gx0 = min(bounds(d)[0] for d in group)
            gy0 = min(bounds(d)[1] for d in group)
            gx1 = max(bounds(d)[2] for d in group)
            gy1 = max(bounds(d)[3] for d in group)
            gcy, gh = (gy0 + gy1) / 2, gy1 - gy0
            gap = max(0.0, max(gx0, x0) - min(gx1, x1))
            if abs(cy - gcy) <= 0.42 * max(height, gh) and gap <= 2.5 * max(height, gh):
                best = group
                break
        if best is None:
            groups.append([det])
        else:
            best.append(det)

    merged = []
    for group in groups:
        if len(group) == 1:
            merged.append(group[0])
            continue
        group = sorted(group, key=lambda d: bounds(d)[0])
        pts = np.concatenate([d.polygon for d in group])
        rect = cv2.minAreaRect(pts.astype(np.float32))
        poly = order_quad(cv2.boxPoints(rect))
        weights = [max(1.0, bounds(d)[2] - bounds(d)[0]) for d in group]
        merged.append(Detection(poly, " ".join(d.text for d in group),
                                float(np.average([d.confidence for d in group], weights=weights))))
    return sorted(merged, key=lambda d: (bounds(d)[1], bounds(d)[0]))


def expanded_crop_transform(poly: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    poly = order_quad(poly)
    width = max(1, round(max(np.linalg.norm(poly[1] - poly[0]), np.linalg.norm(poly[2] - poly[3]))))
    height = max(1, round(max(np.linalg.norm(poly[3] - poly[0]), np.linalg.norm(poly[2] - poly[1]))))
    mx, my = max(12, round(0.06 * width)), max(10, round(0.28 * height))
    canvas = (width + 2 * mx, height + 2 * my)
    dst = np.float32([[mx, my], [mx + width - 1, my],
                      [mx + width - 1, my + height - 1], [mx, my + height - 1]])
    return cv2.getPerspectiveTransform(poly, dst), canvas


def warp_crop(image: np.ndarray, transform: np.ndarray, canvas: tuple[int, int]) -> np.ndarray:
    """Warp a crop and use the photo's median board colour outside its boundary."""
    sample = image[::32, ::32].reshape(-1, 3)
    border = tuple(int(v) for v in np.median(sample, axis=0))
    return cv2.warpPerspective(image, transform, canvas, flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def polygon_inside_fraction(poly: np.ndarray, shape: tuple[int, ...]) -> float:
    h, w = shape[:2]
    area = abs(cv2.contourArea(poly.astype(np.float32)))
    clipped = poly.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, w - 1)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, h - 1)
    return abs(cv2.contourArea(clipped.astype(np.float32))) / max(area, 1.0)


def local_scale(homography: np.ndarray, point: np.ndarray) -> float:
    x, y = map(float, point)
    eps = 10.0
    pts = np.float32([[[x, y], [x + eps, y], [x, y + eps]]])
    mapped = cv2.perspectiveTransform(pts, homography)[0]
    sx = np.linalg.norm(mapped[1] - mapped[0]) / eps
    sy = np.linalg.norm(mapped[2] - mapped[0]) / eps
    return float(math.sqrt(max(sx * sy, 1e-8)))


def sharpness(image: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def ink_correlation(input_crop: np.ndarray, target_crop: np.ndarray) -> float:
    """Compare dark-stroke structure after removing gradual board lighting."""
    a = cv2.cvtColor(input_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(target_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    kernel = max(3, (min(a.shape) // 4) | 1)
    ink_a = np.maximum(cv2.GaussianBlur(a, (kernel, kernel), 0) - a, 0)
    ink_b = np.maximum(cv2.GaussianBlur(b, (kernel, kernel), 0) - b, 0)
    ink_a = (ink_a - ink_a.mean()).ravel()
    ink_b = (ink_b - ink_b.mean()).ravel()
    return float(np.dot(ink_a, ink_b) / (np.linalg.norm(ink_a) * np.linalg.norm(ink_b) + 1e-9))


def resize_pair(input_crop: np.ndarray, target_crop: np.ndarray,
                target_height: int) -> tuple[np.ndarray, np.ndarray]:
    """Optionally resize an aligned pair; zero preserves its native target canvas."""
    if target_height <= 0:
        return input_crop, target_crop
    height, width = target_crop.shape[:2]
    new_width = max(2, round(width * target_height / height))
    size = (new_width, target_height)
    target_resized = cv2.resize(target_crop, size, interpolation=cv2.INTER_AREA)
    input_resized = cv2.resize(input_crop, size, interpolation=cv2.INTER_AREA)
    return input_resized, target_resized


def synthetic_degrade(target: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    factor = int(rng.choice([2, 3, 4], p=[0.30, 0.45, 0.25]))
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


def safe_write(path: Path, image: np.ndarray, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return
    if not cv2.imwrite(str(path), image):
        raise OSError(f"could not write {path}")


def short_hash(*items: str) -> str:
    return hashlib.sha1("|".join(items).encode()).hexdigest()[:10]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_contact_sheet(rows: list[dict], output: Path, limit: int = 24,
                       filename: str = "contact_sheet_real_pairs.jpg") -> None:
    selected = [r for r in rows if r["pair_type"] == "real"][:limit]
    if not selected:
        return
    columns, card_w, image_h, label_h = 3, 700, 190, 62
    rows_count = math.ceil(len(selected) / columns)
    card_h = 2 * image_h + label_h
    sheet = Image.new("RGB", (columns * card_w, rows_count * card_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, row in enumerate(selected):
        column, grid_row = index % columns, index // columns
        x0, y0 = column * card_w, grid_row * card_h
        y = y0
        for label, key in [("REAL INPUT", "input_path"), ("CLEAR TARGET", "target_path")]:
            image = Image.open(output / row[key]).convert("RGB")
            image.thumbnail((card_w - 12, image_h - 4), Image.Resampling.LANCZOS)
            x = x0 + (card_w - image.width) // 2
            sheet.paste(image, (x, y))
            draw.text((x0 + 8, y + 4), label,
                      fill="red" if label.startswith("REAL") else "green", font=font)
            y += image_h
        title = f"{row['input_photo']} -> {row['target_photo']}"
        draft = f"OCR draft: {row['transcription_draft']}"
        draw.text((x0 + 8, y + 6), title[:100], fill="black", font=font)
        draw.text((x0 + 8, y + 24), draft[:100], fill="black", font=font)
        draw.text((x0 + 8, y + 42), "STATUS: NEEDS HUMAN REVIEW", fill="#a00000", font=font)
    sheet.save(output / filename, quality=92)


def main() -> int:
    args = parse_args()
    board_root = args.curated_root / "17oct_board01"
    roles = {r.strip() for r in args.roles.split(",") if r.strip()}
    with (args.curated_root / "pairing_plan.csv").open(newline="", encoding="utf-8") as handle:
        plans = [row for row in csv.DictReader(handle) if row["input_role"] in roles]
    if args.max_inputs > 0:
        plans = plans[: args.max_inputs]
    if not plans:
        raise SystemExit("No pairing-plan rows matched the requested roles.")

    args.output.mkdir(parents=True, exist_ok=True)
    ocr = paddle_detector()
    target_cache: dict[str, tuple[np.ndarray, list[Detection]]] = {}
    manifest: list[dict] = []
    alignments: list[dict] = []
    synthetic_created: set[str] = set()
    rng = np.random.default_rng(args.seed)

    for plan_index, plan in enumerate(plans, 1):
        source_path = locate_photo(board_root, plan["input_file"])
        source = read_heic_bgr(source_path)
        targets = [name for name in plan["target_files"].split(";") if name]
        print(f"[{plan_index}/{len(plans)}] {source_path.name}: {len(targets)} close target(s)")
        for target_name in targets:
            target_path = locate_photo(board_root, target_name)
            if target_name not in target_cache:
                target_image = read_heic_bgr(target_path)
                detections = detect_lines(ocr, target_image)
                target_cache[target_name] = (target_image, detections)
                print(f"  {target_name}: {len(detections)} line candidates")
            target, detections = target_cache[target_name]
            h_target_source, stats = estimate_homography(target, source)
            alignment_id = f"{source_path.stem}__{target_path.stem}"
            accepted = h_target_source is not None
            alignments.append({
                "alignment_id": alignment_id,
                "input_file": source_path.name,
                "target_file": target_path.name,
                "input_role": plan["input_role"],
                **stats,
                "status": "accepted" if accepted else "rejected",
            })
            if not accepted:
                print(f"  {target_name}: alignment rejected ({stats['inliers']} inliers)")
                continue

            # `estimate_homography` already checks this, but keep the extraction
            # stage defensive so a numerically unstable far view is skipped.
            try:
                inv_h = np.linalg.inv(h_target_source)
            except np.linalg.LinAlgError:
                alignments[-1]["status"] = "rejected_singular"
                print(f"  {target_name}: singular alignment rejected")
                continue
            kept = 0
            for line_index, det in enumerate(detections, 1):
                mapped = cv2.perspectiveTransform(det.polygon.reshape(1, -1, 2), h_target_source)[0]
                if polygon_inside_fraction(mapped, source.shape) < 0.92:
                    continue
                crop_h, canvas = expanded_crop_transform(det.polygon)
                target_crop = warp_crop(target, crop_h, canvas)
                source_crop = warp_crop(source, crop_h @ inv_h, canvas)
                if min(target_crop.shape[:2]) < 24 or target_crop.shape[1] < 60:
                    continue
                source_crop, target_crop = resize_pair(source_crop, target_crop, args.target_height)
                pair_id = f"{alignment_id}__L{line_index:02d}_{short_hash(det.text, str(line_index))}"
                target_rel = Path("targets") / f"{target_path.stem}__L{line_index:02d}.png"
                input_rel = Path("inputs_real") / f"{pair_id}.png"
                safe_write(args.output / target_rel, target_crop, args.force)
                safe_write(args.output / input_rel, source_crop, args.force)
                scale = local_scale(h_target_source, det.polygon.mean(axis=0))
                correlation = ink_correlation(source_crop, target_crop)
                geometry_priority = "high" if correlation < 0.70 or stats["inlier_ratio"] < 0.35 else "normal"
                base = {
                    "pair_id": pair_id,
                    "pair_type": "real",
                    "group_id": "17oct_board01",
                    "split": "unassigned",
                    "input_path": input_rel.as_posix(),
                    "target_path": target_rel.as_posix(),
                    "input_photo": source_path.name,
                    "target_photo": target_path.name,
                    "input_role": plan["input_role"],
                    "transcription_draft": det.text,
                    "recognition_confidence": round(det.confidence, 4),
                    "review_status": "needs_review",
                    "geometry_review_priority": geometry_priority,
                    "transcription_review_priority": "high" if det.confidence < 0.80 else "normal",
                    "alignment_inlier_ratio": round(stats["inlier_ratio"], 4),
                    "alignment_median_error_px": round(stats["median_error_px"], 3),
                    "ink_correlation": round(correlation, 4),
                    "target_sharpness": round(sharpness(target_crop), 2),
                    "input_sharpness": round(sharpness(source_crop), 2),
                    "source_pixels_per_target_pixel": round(scale, 4),
                    "suggested_upscale": round(1.0 / max(scale, 1e-6), 2),
                    "degradation": "real_camera",
                }
                manifest.append(base)
                kept += 1

                # Synthetic pairs supplement the real pair; they do not count as new writing.
                synthetic_key = f"{target_path.stem}__L{line_index:02d}"
                if synthetic_key in synthetic_created:
                    continue
                synthetic_created.add(synthetic_key)
                for variant in range(1, args.synthetic_per_target + 1):
                    synthetic, params = synthetic_degrade(target_crop, rng)
                    syn_id = f"{target_path.stem}__L{line_index:02d}__S{variant:02d}"
                    syn_rel = Path("inputs_synthetic") / f"{syn_id}.png"
                    safe_write(args.output / syn_rel, synthetic, args.force)
                    manifest.append({
                        **base,
                        "pair_id": syn_id,
                        "pair_type": "synthetic",
                        "input_path": syn_rel.as_posix(),
                        "input_photo": target_path.name,
                        "input_role": "synthetic_from_close_target",
                        "geometry_review_priority": "synthetic",
                        "alignment_inlier_ratio": "",
                        "alignment_median_error_px": "",
                        "ink_correlation": round(ink_correlation(synthetic, target_crop), 4),
                        "input_sharpness": round(sharpness(synthetic), 2),
                        "source_pixels_per_target_pixel": round(1.0 / params["downscale"], 4),
                        "suggested_upscale": params["downscale"],
                        "degradation": ";".join(f"{k}={v}" for k, v in params.items()),
                    })
            print(f"  {target_name}: {kept} real line pairs kept; "
                  f"inliers={stats['inliers']} ({stats['inlier_ratio']:.0%})")

        # Save progress after every input photograph. A later difficult far
        # view cannot erase already prepared pairs if the run is interrupted.
        write_csv(args.output / "manifest.csv", manifest)
        write_csv(args.output / "alignment_summary.csv", alignments)

    write_csv(args.output / "manifest.csv", manifest)
    write_csv(args.output / "alignment_summary.csv", alignments)
    make_contact_sheet(manifest, args.output)
    real_count = sum(r["pair_type"] == "real" for r in manifest)
    synthetic_count = len(manifest) - real_count
    print(f"\nCreated {real_count} real and {synthetic_count} synthetic candidate pairs.")
    print(f"Review: {args.output / 'contact_sheet_real_pairs.jpg'}")
    print(f"Edit labels/status: {args.output / 'manifest.csv'}")
    return 0 if real_count else 2


if __name__ == "__main__":
    sys.exit(main())
