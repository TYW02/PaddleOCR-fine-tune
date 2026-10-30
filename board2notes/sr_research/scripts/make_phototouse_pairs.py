#!/usr/bin/env python3
"""Pairs from the PhotosToUse folder (teammates' robustness photos) for the super-resolution experiment.

PhotosToUse holds four kinds of image (see docs/SUPER_RESOLUTION_RUNS.md):
  * OnePlus whiteboard photos (1000092181-211): blue marker on a whiteboard, several shots per text,
    all sharp and large (letters 130-350 px tall)
  * named line crops "<text><variant>.jpg": 11 whiteboard texts, variants dim / rotation / angle (clear)
    and blur (REAL camera shake, confirmed by the team)
  * iPhone pencil-on-paper photos and the "Singapore 25 May" paper photos: NOT used here (different ink
    and surface; kept for a later whiteboard -> paper transfer demo)

Three outputs, written to outputs/paired_dataset_phototouse/. Train and test never share a text:

1. train_synth (D3)  TRAINING pairs. Writing regions of the OnePlus photos, scaled so the letters are as
   big as the Airdrop targets (median ink-blob height TARGET_BLOB_H), degraded with the same recipe as
   make_airdrop_synthetic_pairs.py. Photos showing "No food" / "Jun hong is wrong about ronaldo" are left
   out because those texts are in the tests; 1000092201 is too dark to read.

2. lowres_test (D4)  SECONDARY TEST, synthetic, resolution only. Each named text is straightened and cut
   into single lines (the PaddleOCR recogniser reads one line), scaled to the Airdrop target size, then
   shrunk by the real far-crop range (source pixels per target pixel 0.22 and 0.35, i.e. the 3.3-6.7x and
   <3.3x bands) with area averaging and enlarged back with bicubic, exactly like a far crop before the
   model. No blur, noise or JPEG is added, so this differs from the training recipe on purpose.

3. shake_test (A3)  STRESS TEST. The real camera-shake crops of single-line texts, trimmed to the smeared
   writing and scaled so the writing WIDTH matches the clear line (vertical shake stretches the height),
   centred on a canvas of the clear line's size. NOT pixel-aligned: use CER only, not PSNR. Motion blur is
   a different degradation from low resolution; the model was never trained for it.

Each manifest has the same columns as paired_dataset_17oct/accepted_manifest.csv. Previews of every
target / input are written to previews/ for a visual check.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pair_data  # noqa: E402
from make_airdrop_synthetic_pairs import VARIANTS, synthetic_degrade  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PHOTOS = REPO.parent / "PhotosToUse"
OUT = REPO / "outputs" / "paired_dataset_phototouse"
TEMPLATE = REPO / "outputs" / "paired_dataset_17oct" / "accepted_manifest.csv"

TARGET_BLOB_H = 25          # median ink-blob height of the Airdrop targets (23-26 px): the test scale
LOWRES_SPP = (0.22, 0.35)   # two enlargement bands of the real Airdrop far crops (0.21-0.39 measured)

# --- which OnePlus photos are training material (texts that are NOT in the tests) ---
TRAIN_PHOTOS = {
    "1000092181": "Reach out for support", "1000092182": "Reach out for support",
    "1000092183": "Reach out for support", "1000092184": "Reach out for support",
    "1000092185": "Reach out for support", "1000092186": "kindly clean the whiteboard",
    "1000092187": "kindly clean the whiteboard", "1000092188": "Looking to take things to a whole new level?",
    "1000092189": "Looking to take things to a whole new level?",
    "1000092190": "Looking to take things to a whole new level?",
    "1000092191": "Looking to take things to a whole new level?", "1000092208": "Ronaldo left portugal training camp",
    "1000092209": "Firefighting Access Do not Obstruct", "1000092210": "AV IT and",
    "1000092211": "I have a dream that yong chang got together with bobi",
}
EXCLUDED_PHOTOS = {
    **{f"10000921{n}": "same text as a test line ('No food')" for n in range(92, 98)},
    **{f"10000922{n:02d}": "same text as a test line ('Jun hong is wrong about ronaldo')" for n in (4, 5, 6)},
    "1000092201": "too dark to read",
}

# --- named crops: one entry per text; lines in reading order (human transcription) ---
NAMED_TEXTS = {
    "Getupto600insavings": ["GET UP TO $600 in Savings"],
    "isaftershockgivingfakegpu": ["is Aftershock giving fake gpu?"],
    "itisokaynottobeokay": ["it is Okay not to be Okay"],
    "junhongiswrongaboutronaldo": ["Jun hong is wrong about ronaldo"],
    "messiretiredfrominternationalfootball": ["Messi retired", "from international", "football"],
    "nikeisbetterthanaddidas": ["Nike is better than addidas"],
    "nofood": ["No food"],
    "onlypentalbrand": ["Only Pental brand"],
    "ronaldoisbetterthanmessiheishim": ["Ronaldo is better than Messi,", "He is him!"],
    "ronaldoisthegoat": ["Ronaldo is the goat"],
    "whiteboardmarkersaretobeused": ["Whiteboard markers", "are to be used"],
}
CLEAR_PREFERENCE = ("angle", "dim", "rotation")   # first available clear variant is the source


# ------------------------------------------------------------------------------------------------
def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)        # applies the EXIF orientation
    if image is None:
        raise FileNotFoundError(path)
    return image


def grey_norm(bgr: np.ndarray) -> np.ndarray:
    return pair_data.normalise_board(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).mean(axis=2)


def blob_height(bgr: np.ndarray, work_side: int = 1500) -> float:
    """Median height (px, at the image's own scale) of the ink blobs: a size measure of the handwriting
    that does not depend on margins. Measured on a copy at most `work_side` px long."""
    s = min(1.0, work_side / max(bgr.shape[:2]))
    small = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else bgr
    ink = (grey_norm(small) < 0.75).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    min_area = max(8, 0.0003 * ink.size)
    heights = [stats[i, 3] for i in range(1, n) if stats[i, 4] >= min_area]
    if not heights:
        raise ValueError("no ink found")
    return float(np.median(heights)) / s


def ink_box(grey: np.ndarray, lo: float = 1, hi: float = 99):
    ys, xs = np.nonzero(cv2.medianBlur((grey < 0.75).astype(np.uint8), 3))
    if len(xs) < 20:
        raise ValueError("no ink found")
    return (int(np.percentile(xs, lo)), int(np.percentile(ys, lo)),
            int(np.percentile(xs, hi)) + 1, int(np.percentile(ys, hi)) + 1)


def crop_box(img, box, margin):
    x0, y0, x1, y1 = box
    h, w = img.shape[:2]
    return img[max(0, y0 - margin):min(h, y1 + margin), max(0, x0 - margin):min(w, x1 + margin)]


def board_colour(bgr):
    return tuple(int(v) for v in np.median(bgr.reshape(-1, 3), axis=0))


def rotate(bgr, angle):
    """Rotate about the centre with an enlarged canvas; corners filled with ONE exact board colour so
    pair_data.valid_mask leaves them out of the metrics."""
    h, w = bgr.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    c, s = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(bgr, m, (nw, nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=board_colour(bgr))


def deskew_angle(bgr) -> float:
    """Angle that makes the text lines horizontal: the rotation whose row profile of ink is most peaked
    (classic projection-profile deskew; works for several lines at once)."""
    ink = (grey_norm(bgr) < 0.75).astype(np.float32)
    best, best_score = 0.0, -1.0
    for coarse in (np.arange(-60, 61, 2.0), None):
        angles = coarse if coarse is not None else np.arange(best - 2, best + 2.01, 0.25)
        for a in angles:
            h, w = ink.shape
            m = cv2.getRotationMatrix2D((w / 2, h / 2), float(a), 1.0)
            r = cv2.warpAffine(ink, m, (w, h), flags=cv2.INTER_NEAREST)
            score = float(np.var(r.sum(axis=1)))
            if score > best_score:
                best, best_score = float(a), score
    return best


def split_lines(bgr, n_lines: int):
    """Cut a straightened multi-line crop into n_lines horizontal bands at the emptiest rows."""
    profile = (grey_norm(bgr) < 0.75).sum(axis=1).astype(np.float32)
    kernel = np.exp(-0.5 * (np.arange(-9, 10) / 3.0) ** 2)
    profile = np.convolve(profile, kernel / kernel.sum(), mode="same")
    ys = np.nonzero(profile > 0.15 * profile.max())[0]
    top, bottom = ys.min(), ys.max() + 1
    if n_lines == 1:
        return [(top, bottom)]
    cuts, start = [], top
    # choose the n_lines-1 lowest local minima between top and bottom, spaced at least 1/(2n) apart
    span = bottom - top
    order = np.argsort(profile[top:bottom]) + top
    chosen = []
    for y in order:
        if y - top < span * 0.15 or bottom - y < span * 0.15:
            continue
        if all(abs(y - c) > span / (2 * n_lines) for c in chosen):
            chosen.append(int(y))
        if len(chosen) == n_lines - 1:
            break
    edges = [top] + sorted(chosen) + [bottom]
    for a, b in zip(edges[:-1], edges[1:]):
        cuts.append((a, b))
    return cuts


def scale_to_target(bgr):
    """Scale so the handwriting matches the Airdrop targets; never enlarge (a blown-up target is soft)."""
    s = min(1.0, TARGET_BLOB_H / blob_height(bgr))
    if s < 1.0:
        bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return bgr, s


def write_manifest(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def preview(path, images, titles):
    """Stack images (BGR) vertically with a title strip, for a quick visual check."""
    width = max(i.shape[1] for i in images)
    strips = []
    for img, title in zip(images, titles):
        canvas = np.full((img.shape[0] + 22, width, 3), 255, np.uint8)
        canvas[22:, :img.shape[1]] = img
        cv2.putText(canvas, title, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1, cv2.LINE_AA)
        strips.append(canvas)
    cv2.imwrite(str(path), np.vstack(strips))


def base_row(columns, **values):
    row = {c: "" for c in columns}
    row.update({"review_status": "accepted", "annotation_source": "human (PhotosToUse file names / photos)"})
    row.update(values)
    return row


# ------------------------------------------------------------------------------------------------
def make_train_synth(columns):
    out = OUT / "train_synth"
    (out / "targets").mkdir(parents=True, exist_ok=True)
    (out / "inputs_synthetic").mkdir(parents=True, exist_ok=True)
    rows, previews = [], []
    for photo, text in sorted(TRAIN_PHOTOS.items()):
        bgr = read_bgr(PHOTOS / f"{photo}.jpg")
        scaled, s = scale_to_target(bgr)
        target = crop_box(scaled, ink_box(grey_norm(scaled)), margin=2 * TARGET_BLOB_H)
        tgt_name = f"targets/{photo}.png"
        cv2.imwrite(str(out / tgt_name), target)
        rng = np.random.default_rng(int(photo) % (2 ** 32))
        for k in range(1, VARIANTS + 1):
            degraded, params = synthetic_degrade(target, rng)
            pair_id = f"{photo}__S{k:02d}"
            in_name = f"inputs_synthetic/{pair_id}.png"
            cv2.imwrite(str(out / in_name), degraded)
            if k == 1:
                previews.append((target, degraded, photo))
            rows.append(base_row(
                columns, pair_id=pair_id, pair_type="synthetic", group_id="phototouse_whiteboard", split="train",
                input_path=str((out / in_name).relative_to(REPO)), target_path=str((out / tgt_name).relative_to(REPO)),
                input_photo=f"{photo}.jpg", target_photo=f"{photo}.jpg", input_role="synthetic_from_close_target",
                source_pixels_per_target_pixel=round(1 / params["downscale"], 4), suggested_upscale=params["downscale"],
                degradation=";".join(f"{a}={b}" for a, b in params.items()), source_set="phototouse_oneplus_synthetic",
                cer_eligible="no", review_notes=f"writing region (several lines) of: {text}; photo scaled x{s:.3f}"))
    write_manifest(out / "phototouse_train_manifest.csv", rows, columns)
    (OUT / "previews").mkdir(exist_ok=True)
    for target, degraded, photo in previews:
        preview(OUT / "previews" / f"train_{photo}.png", [target, degraded], [f"{photo} target", "synthetic input S01"])
    return rows


def clear_source(base):
    for variant in CLEAR_PREFERENCE:
        for ext in ("jpg", "png"):
            path = PHOTOS / f"{base}{variant}.{ext}"
            if path.exists():
                return path, variant
    raise FileNotFoundError(base)


def named_lines():
    """Yield (base, line_index, label, straightened clear line crop BGR at the target scale, variant)."""
    for base, labels in NAMED_TEXTS.items():
        path, variant = clear_source(base)
        bgr = read_bgr(path)
        angle = deskew_angle(bgr)
        if abs(angle) > 1.0:
            bgr = rotate(bgr, angle)
        scaled, _ = scale_to_target(bgr)
        grey = grey_norm(scaled)
        x0, _, x1, _ = ink_box(grey)
        bands = split_lines(scaled, len(labels))
        for i, ((top, bottom), label) in enumerate(zip(bands, labels), start=1):
            band = scaled[top:bottom]
            bx0, by0, bx1, by1 = ink_box(grey_norm(band), lo=0.5, hi=99.5)
            line_h = by1 - by0
            margin_y, margin_x = max(4, int(0.35 * line_h)), max(6, int(0.5 * line_h))
            y0 = max(0, top + by0 - margin_y); y1 = min(scaled.shape[0], top + by1 + margin_y)
            xa = max(0, bx0 - margin_x); xb = min(scaled.shape[1], bx1 + margin_x)
            yield base, i, label, scaled[y0:y1, xa:xb], variant, angle


def make_lowres_test(columns):
    out = OUT / "lowres_test"
    (out / "targets").mkdir(parents=True, exist_ok=True)
    (out / "inputs_lowres").mkdir(parents=True, exist_ok=True)
    rows, previews, targets = [], [], {}
    for base, i, label, target, variant, angle in named_lines():
        tgt_name = f"targets/{base}_L{i:02d}.png"
        cv2.imwrite(str(out / tgt_name), target)
        targets[(base, i)] = (out / tgt_name, label, variant)
        h, w = target.shape[:2]
        shown = [target]
        for spp in LOWRES_SPP:
            low = cv2.resize(target, (max(2, round(w * spp)), max(2, round(h * spp))), interpolation=cv2.INTER_AREA)
            up = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
            pair_id = f"{base}_L{i:02d}__spp{int(spp * 100):02d}"
            in_name = f"inputs_lowres/{pair_id}.png"
            cv2.imwrite(str(out / in_name), up)
            shown.append(up)
            rows.append(base_row(
                columns, pair_id=pair_id, pair_type="synthetic_lowres", group_id="phototouse_named", split="test",
                input_path=str((out / in_name).relative_to(REPO)), target_path=str((out / tgt_name).relative_to(REPO)),
                input_photo=f"{base}{variant}", target_photo=f"{base}{variant}", input_role="downscaled_clear",
                source_pixels_per_target_pixel=spp, suggested_upscale=round(1 / spp, 2),
                degradation=f"area_downscale={spp};bicubic_up", source_set="phototouse_named_lowres",
                transcription_verified=label, cer_eligible="yes",
                review_notes=f"clear '{variant}' crop, straightened by {angle:.1f} deg, line {i}"))
        previews.append((shown, f"{base} L{i} ({variant}): {label}"))
    write_manifest(out / "phototouse_lowres_test_manifest.csv", rows, columns)
    for k, (shown, title) in enumerate(previews):
        preview(OUT / "previews" / f"lowres_{k:02d}.png", shown,
                [title, f"input spp {LOWRES_SPP[0]}", f"input spp {LOWRES_SPP[1]}"])
    return rows, targets


def trim_to_content(bgr, keep: float = 0.25, pad: float = 0.15):
    """Crop a shaken crop to its smeared writing. Smeared ink is too faint for the 0.75 ink threshold,
    so rows / columns are kept where the mean darkness is above `keep` x its maximum."""
    dark = 1.0 - grey_norm(bgr)
    def span(profile):
        profile = np.convolve(profile, np.ones(5) / 5, mode="same")
        idx = np.nonzero(profile > keep * profile.max())[0]
        return idx.min(), idx.max() + 1
    y0, y1 = span(dark.mean(axis=1))
    x0, x1 = span(dark.mean(axis=0))
    py, px = int(pad * (y1 - y0)), int(pad * (y1 - y0))
    return bgr[max(0, y0 - py):y1 + py, max(0, x0 - px):x1 + px]


def make_shake_test(columns, targets):
    out = OUT / "shake_test"
    (out / "inputs_shake").mkdir(parents=True, exist_ok=True)
    rows, rejected = [], []
    for base, labels in NAMED_TEXTS.items():
        if len(labels) != 1:
            rejected.append((base, "several lines: the line recogniser cannot read the shaken block"))
            continue
        blur_path = PHOTOS / f"{base}blur.jpg"
        if not blur_path.exists():
            rejected.append((base, "no blur crop")); continue
        tgt_path, label, variant = targets[(base, 1)]
        target = cv2.imread(str(tgt_path))
        shaken = trim_to_content(read_bgr(blur_path))
        # Shake smears the writing along one direction (here vertically), so the shaken block is taller
        # than the letters and ECC rarely locks on. Scale so the WRITING WIDTH matches the clear line
        # (width is not stretched by vertical shake) and centre it on a canvas of the target's size.
        th, tw = target.shape[:2]
        tx0, _, tx1, _ = ink_box(grey_norm(target))
        s = (tx1 - tx0) / max(1, shaken.shape[1] - 2 * int(0.15 * shaken.shape[0]))
        up = cv2.resize(shaken, (max(1, round(shaken.shape[1] * s)), max(1, round(shaken.shape[0] * s))),
                        interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
        canvas = np.empty_like(target)
        canvas[:] = board_colour(shaken)
        uh, uw = up.shape[:2]
        sy, sx = max(0, (uh - th) // 2), max(0, (uw - tw) // 2)          # crop overflow
        dy, dx = max(0, (th - uh) // 2), max(0, (tw - uw) // 2)          # or pad
        ch, cw = min(th, uh), min(tw, uw)
        canvas[dy:dy + ch, dx:dx + cw] = up[sy:sy + ch, sx:sx + cw]
        aligned_bgr, note = canvas, f"writing width matched (x{s:.2f}) and centred, not pixel-aligned"
        pair_id = f"{base}__shake"
        in_name = f"inputs_shake/{pair_id}.png"
        cv2.imwrite(str(out / in_name), aligned_bgr)
        rows.append(base_row(
            columns, pair_id=pair_id, pair_type="real_shake", group_id="phototouse_shake", split="test",
            input_path=str((out / in_name).relative_to(REPO)), target_path=str(tgt_path.relative_to(REPO)),
            input_photo=blur_path.name, target_photo=f"{base}{variant}", input_role="camera_shake",
            source_pixels_per_target_pixel=round(shaken.shape[0] / target.shape[0], 4),
            degradation="real_camera_shake", source_set="phototouse_shake", transcription_verified=label,
            cer_eligible="yes", review_notes=note + "; use CER only, PSNR is unreliable for shaken text"))
        preview(OUT / "previews" / f"shake_{base}.png", [aligned_bgr, target], [f"shaken: {note}", f"clear: {label}"])
    write_manifest(out / "phototouse_shake_test_manifest.csv", rows, columns)
    return rows, rejected


def main() -> None:
    with TEMPLATE.open(newline="", encoding="utf-8") as fh:
        columns = list(csv.DictReader(fh).fieldnames)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "previews").mkdir(exist_ok=True)
    train = make_train_synth(columns)
    lowres, targets = make_lowres_test(columns)
    shake, rejected = make_shake_test(columns, targets)
    test_texts = {r["transcription_verified"].lower() for r in lowres + shake}
    overlap = [t for t in TRAIN_PHOTOS.values() if t.lower() in test_texts]
    assert not overlap, f"train and test share texts: {overlap}"
    print(f"D3 training pairs : {len(train)} from {len(TRAIN_PHOTOS)} OnePlus photos "
          f"({len(set(TRAIN_PHOTOS.values()))} texts); excluded {len(EXCLUDED_PHOTOS)} photos")
    print(f"D4 low-res test   : {len(lowres)} pairs from {len(targets)} lines of {len(NAMED_TEXTS)} texts")
    print(f"A3 shake test     : {len(shake)} pairs; skipped: " + "; ".join(f"{b} ({r})" for b, r in rejected))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
