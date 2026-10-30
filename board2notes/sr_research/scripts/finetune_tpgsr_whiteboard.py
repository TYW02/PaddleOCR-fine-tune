#!/usr/bin/env python3
"""Fine-tune the TextZoom-trained TPGSR on our whiteboard WORD crops, then test on a board it never saw.

Question: does a text super-resolution model that already "knows characters" (pretrained on ~17k
TextZoom text pairs, guided by a CRNN text prior) help PaddleOCR read far-away handwriting once it is
fine-tuned on our marker handwriting?

Steps (python scripts/finetune_tpgsr_whiteboard.py <step>):
  prepare   cut every far/close line pair into word pairs (LR 16x64 from the FAR photo, HR 32x128 from
            the close photo). Word boxes come from the far photo only, as they would in the app.
  prior     no training: how well does the frozen CRNN (the source of the "text prior") read our LR words?
            If it misreads them, the prior tells TPGSR to draw the WRONG letters clearly.
  train     fine-tune outputs/tpgsr_textzoom/tpgsr_soft_prior_best.pt; every batch mixes our words with
            TextZoom words (so it does not forget TextZoom); the best epoch is chosen by PaddleOCR CER on
            the validation LINES (epoch 0 = the pretrained model, so fine-tuning has to beat it).
  evaluate  restore every word, paste it back into its line, and let PaddleOCR read the line. Compared on
            the same lines: raw input | 16px path, no network | TPGSR pretrained | TPGSR fine-tuned |
            blend | confidence gate. Blend weight and gate margin are chosen on validation, then the
            test set is read once. Main test: Airdrop B3 (CNN board), never used for training.
  all       prepare + prior + train + evaluate

Size rules (from the word-swap test, scripts/word_swap_counterfactual.py, 20 Oct): on B3, blur only costs
PaddleOCR letters in words under ~16-20 px (far-photo pixels); taller words read as well as the clear photo.
TPGSR's output is 32 px tall, so a taller word can only LOSE detail on the 16 px route. Hence:
  - only words between min_native_h and sr_max_h px are trained on, validated on and restored;
  - at test time a size gate (chosen on validation) restores only words below its height, the rest stay raw.
  --smoke   tiny run (a few lines, 2 short epochs) to check that everything works (~2-4 min).

The CER is the team's line CER (lower case, single spaces, capped at 1), so `raw input` on B3 should be
about 0.153: the 0.157 of runs 1-4 / M1 / M2 on 24 lines, minus the one line with no word box (word-swap test).
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/whiteboard-ocr-matplotlib")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
import pair_data  # noqa: E402

OUT = REPO / "outputs" / "tpgsr_whiteboard"
BASE_CHECKPOINT = REPO / "outputs" / "tpgsr_textzoom" / "tpgsr_soft_prior_best.pt"
BEST_CHECKPOINT = REPO / "models" / "superres" / "tpgsr_whiteboard_best.pt"
MANIFESTS = {   # name -> (manifest, how its rows are used)
    "17oct": REPO / "outputs" / "paired_dataset_17oct" / "accepted_split_manifest.csv",     # train / val / test split column
    "airdrop_synth": REPO / "outputs" / "paired_dataset_airdrop_synth" / "airdrop_synth_manifest.csv",
    "airdrop_real": REPO / "outputs" / "paired_dataset_airdrop" / "airdrop_test_manifest.csv",
}
TEST_GROUP = "airdrop_B3"          # the CNN board: never in training, never in validation
LR_SIZE, HR_SIZE = (64, 16), (128, 32)   # (width, height) as in TextZoom


@dataclass
class Settings:
    # word splitting (fractions of the line's ink height)
    gap_frac: float = 0.40          # a gap at least this wide separates two words
    margin_frac: float = 0.15       # board kept around each word
    max_aspect: float = 6.0         # longer pieces are split at their emptiest column (TextZoom words are short)
    # data
    synth_val_frac: float = 0.15    # Airdrop synthetic LINES (all their degraded copies) held out for validation
    min_native_h: float = 10.0      # TRAIN only on words whose box in the far photo is >= this many real pixels tall.
                                    # Lecture 8, slide 17: an augmentation is valid only if a human would keep the label.
                                    # 10 px, not 16: in the word-swap figure B3 words of 9-12 px are still readable by eye,
                                    # and words of 13-16 px are where fixing blur helps most. Validation/test keep every word.
    sr_max_h: float = 32.0          # never train on / restore words taller than this (TPGSR output is 32 px tall)
    size_gate_choices: tuple = (16.0, 20.0, 24.0, 32.0)   # test-time size gate, chosen on validation
    # training
    epochs: int = 10
    steps_per_epoch: int = 200
    our_per_batch: int = 8          # our word pairs per batch
    textzoom_per_batch: int = 8     # TextZoom pairs per batch (0 = none)
    learning_rate: float = 5e-5     # lower than training from scratch (1e-4): fine-tuning
    gradient_weight: float = 0.20   # same loss as the TextZoom training
    prior_weight: float = 0.01
    patience: int = 4               # epochs without a better validation CER -> stop
    seed: int = 20261020
    smoke: bool = False


# ----------------------------------------------------------------------------------------------- text
def norm_text(t: str) -> str:
    return " ".join(str(t).lower().split())


def line_cer(reference: str, prediction: str) -> float:
    import jiwer
    return min(float(jiwer.cer(norm_text(reference), norm_text(prediction) or " ")), 1.0)


def label_of(row: dict) -> str:
    return (row.get("transcription_verified") or row.get("transcription_draft") or "").strip()


# ----------------------------------------------------------------------------------------------- words
def word_boxes(rgb: np.ndarray, ok: np.ndarray, s: Settings) -> list[tuple[int, int, int, int]]:
    """Word boxes (x0, y0, x1, y1) of one line image, from its own ink (the FAR photo at test time).

    1. Ink = clearly darker than the board. The threshold adapts to the line (far photos are pale), and
       faint ghost writing from earlier erasing stays out.
    2. The line's own band = the run of rows holding most ink; fragments of the lines above / below
       (cut into the canvas) are ignored.
    3. Words = column runs of ink separated by gaps of at least gap_frac x band height; pieces longer than
       max_aspect are split at their emptiest column (TextZoom words are short)."""
    norm = pair_data.normalise_board(rgb).mean(axis=2)
    vals = norm[ok]
    if vals.size < 100:
        return []
    dark = float(np.percentile(vals, 1.0))
    if dark > 0.85:                                     # no real ink
        return []
    thresh = 1.0 - 0.45 * (1.0 - dark)                  # e.g. darkest 0.40 -> ink below 0.73; darkest 0.10 -> 0.60
    ink = ((norm < thresh) & ok).astype(np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    rows = cv2.GaussianBlur(ink.sum(axis=1).astype(np.float32)[:, None], (1, 0), sigmaX=1, sigmaY=3).ravel()
    if rows.max() <= 0:
        return []
    peak = int(np.argmax(rows))
    on = rows > 0.15 * rows.max()
    y0 = peak
    while y0 > 0 and on[y0 - 1]:
        y0 -= 1
    y1 = peak + 1
    while y1 < len(on) and on[y1]:
        y1 += 1
    h = y1 - y0
    if h < 4:
        return []
    band = ink[y0:y1]
    colsum = band.sum(axis=0)
    xs = np.nonzero(colsum > 0)[0]
    if not len(xs):
        return []
    segs, start, prev = [], int(xs[0]), int(xs[0])
    for x in xs[1:]:
        if x - prev - 1 >= s.gap_frac * h:
            segs.append([start, prev + 1])
            start = int(x)
        prev = int(x)
    segs.append([start, prev + 1])
    segs = [g for g in segs if colsum[g[0]:g[1]].sum() >= 0.03 * h * h]            # drop specks / dots

    def split(g):
        a, b = g
        if b - a <= s.max_aspect * h:
            return [g]
        lo, hi = a + int(0.35 * (b - a)), a + int(0.65 * (b - a))
        cut = lo + int(np.argmin(colsum[lo:hi]))
        return split([a, cut]) + split([cut, b])

    segs = [p for g in segs for p in split(g)]
    m = int(round(s.margin_frac * h))
    H, W = ink.shape
    boxes = []
    for i, (a, b) in enumerate(segs):
        left = a - m if i == 0 else max(a - m, (segs[i - 1][1] + a) // 2)        # never overlap the neighbour
        right = b + m if i == len(segs) - 1 else min(b + m, (b + segs[i + 1][0]) // 2)
        boxes.append((max(0, left), max(0, y0 - m), min(W, right), min(H, y1 + m)))
    return [bx for bx in boxes if bx[2] - bx[0] >= 4 and bx[3] - bx[1] >= 4]


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def manifest_rows(s: Settings) -> list[dict]:
    """Every usable line pair with its split: train / val / test."""
    rows = []
    for name, path in MANIFESTS.items():
        if not path.exists():
            print(f"  missing {path.relative_to(REPO)}: skipped")
            continue
        for r in csv.DictReader(open(path, encoding="utf-8")):
            r = dict(r, source=name)
            if name == "17oct":
                r["split_use"] = r.get("split", "train")
            elif name == "airdrop_real":
                r["split_use"] = "test" if r["group_id"] == TEST_GROUP else "val"     # B2: validation
            else:
                r["split_use"] = "train"
            rows.append(r)
    # Airdrop synthetic: hold out whole target lines (all their degraded copies) for validation
    synth = sorted({r["target_path"] for r in rows if r["source"] == "airdrop_synth"})
    held = set(random.Random(s.seed).sample(synth, int(round(s.synth_val_frac * len(synth))))) if synth else set()
    for r in rows:
        if r["source"] == "airdrop_synth" and r["target_path"] in held:
            r["split_use"] = "val"
    assert not any(r["group_id"] == TEST_GROUP and r["split_use"] != "test" for r in rows), "test board leaked"
    return rows


def prepare(s: Settings) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = manifest_rows(s)
    if s.smoke:   # a few lines of every kind, always including the main test board
        rows = ([r for r in rows if r["split_use"] == "train" and r["source"] == "17oct"][:6]
                + [r for r in rows if r["split_use"] == "train" and r["source"] == "airdrop_synth"][:6]
                + [r for r in rows if r["split_use"] == "val" and r["source"] == "17oct"][:3]
                + [r for r in rows if r["split_use"] == "val" and r["source"] == "airdrop_synth"][:3]
                + [r for r in rows if r["group_id"] == TEST_GROUP][:4]
                + [r for r in rows if r["split_use"] == "test" and r["group_id"] != TEST_GROUP][:2])
    lines, lr_words, hr_words, words = [], [], [], []
    t0 = time.time()
    for k, r in enumerate(rows):
        inp, tgt = read_rgb(REPO / r["input_path"]), read_rgb(REPO / r["target_path"])
        if inp.shape != tgt.shape:
            continue
        ok = pair_data.valid_mask(inp) & pair_data.valid_mask(tgt)
        boxes = []
        try:
            spp = float(r.get("source_pixels_per_target_pixel") or "nan")
        except ValueError:
            spp = float("nan")
        for bx in word_boxes(inp, ok, s):
            x0, y0, x1, y1 = bx
            if ok[y0:y1, x0:x1].mean() < 0.9:
                continue
            lr_words.append(cv2.resize(inp[y0:y1, x0:x1], LR_SIZE, interpolation=cv2.INTER_AREA))
            hr_words.append(cv2.resize(tgt[y0:y1, x0:x1], HR_SIZE, interpolation=cv2.INTER_AREA))
            native_h = (y1 - y0) * spp if spp == spp else float(y1 - y0)       # real pixels in the far photo
            words.append(dict(word=len(words), line=len(lines), split=r["split_use"], x0=x0, y0=y0, x1=x1, y1=y1,
                              native_h=round(native_h, 1)))
            boxes.append(len(words) - 1)
        lines.append(dict(line=len(lines), pair_id=r["pair_id"], source=r["source"], group=r["group_id"],
                          split=r["split_use"], pair_type=r.get("pair_type", ""), label=label_of(r),
                          cer_eligible=r.get("cer_eligible", "yes"), input_path=r["input_path"],
                          target_path=r["target_path"], words=boxes))
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(rows)} lines, {len(words)} words ({time.time() - t0:.0f} s)")
    np.savez_compressed(OUT / "word_pairs.npz", lr=np.stack(lr_words), hr=np.stack(hr_words))
    json.dump(dict(settings=asdict(s), lines=lines), open(OUT / "lines.json", "w"), indent=1)
    with open(OUT / "words.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(words[0].keys()))
        w.writeheader(); w.writerows(words)
    summary = {}
    for ln in lines:
        key = (ln["split"], ln["source"] if ln["source"] != "airdrop_real" else ln["group"])
        a = summary.setdefault(key, [0, 0])
        a[0] += 1; a[1] += len(ln["words"])
    print("lines / words per split:")
    for (split, src), (n_l, n_w) in sorted(summary.items()):
        print(f"  {split:<5} {src:<14} {n_l:>4} lines {n_w:>5} words")
    print(f"saved {len(words)} word pairs -> {OUT.relative_to(REPO)}")
    preview_words(lines, np.stack(lr_words), np.stack(hr_words))


def preview_words(lines, lr, hr, n=6):
    """Check the word splitting by eye: boxes on the far line, and a few LR/HR word pairs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pick = ([ln for ln in lines if ln["group"] == TEST_GROUP][::6][:4] + [ln for ln in lines if ln["source"] == "17oct"][::60][:3]
            + [ln for ln in lines if ln["source"] == "airdrop_synth"][::80][:2])
    fig, axes = plt.subplots(len(pick) + 2, 1, figsize=(12, 2.2 * (len(pick) + 2)))
    for a, ln in zip(axes, pick):
        img = read_rgb(REPO / ln["input_path"])
        a.imshow(img)
        for wi in ln["words"]:
            w = WORDS_CACHE(wi)
            a.add_patch(plt.Rectangle((w["x0"], w["y0"]), w["x1"] - w["x0"], w["y1"] - w["y0"], fill=False, ec="red", lw=1))
        a.set_title(f"{ln['split']} | {ln['label']!r} | {len(ln['words'])} word boxes (from the far photo)", fontsize=8, loc="left")
        a.axis("off")
    idx = np.linspace(0, len(lr) - 1, n).astype(int)
    axes[-2].imshow(np.hstack([cv2.resize(lr[i], HR_SIZE, interpolation=cv2.INTER_NEAREST) for i in idx]))
    axes[-2].set_title("LR word crops (16 x 64, shown enlarged): the model's input", fontsize=8, loc="left")
    axes[-1].imshow(np.hstack([hr[i] for i in idx]))
    axes[-1].set_title("HR word crops (32 x 128) from the close photo: the target", fontsize=8, loc="left")
    for a in axes[-2:]:
        a.axis("off")
    plt.tight_layout()
    fig.savefig(OUT / "word_split_preview.png", dpi=130)
    plt.close(fig)
    print(f"check the word splitting: {(OUT / 'word_split_preview.png').relative_to(REPO)}")


_WORDS = None


def WORDS_CACHE(i):
    global _WORDS
    if _WORDS is None:
        _WORDS = list(csv.DictReader(open(OUT / "words.csv")))
        conv = {"split": str, "native_h": float}
        _WORDS = [{k: conv.get(k, int)(v) for k, v in w.items()} for w in _WORDS]
    return _WORDS[i]


def load_prepared():
    z = np.load(OUT / "word_pairs.npz")
    meta = json.load(open(OUT / "lines.json"))
    words = [WORDS_CACHE(i) for i in range(len(z["lr"]))]
    return z["lr"], z["hr"], meta["lines"], words


# ----------------------------------------------------------------------------------------------- model
def models(device):
    import train_tpgsr_textzoom as tz
    source = REPO.parent / "models" / "TPGSR"
    return tz, tz.build_generator(source, device), tz.build_recognizer(source, device)


def to_tensor(batch_uint8):
    import torch
    return torch.from_numpy(np.ascontiguousarray(batch_uint8, dtype=np.float32) / 255.0).permute(0, 3, 1, 2)


def restore(tz, generator, recognizer, lr_uint8, device, bs=64):
    """LR words (N, 16, 64, 3) uint8 -> restored HR words (N, 32, 128, 3) uint8."""
    import torch
    generator.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(lr_uint8), bs):
            lr = to_tensor(lr_uint8[i:i + bs]).to(device)
            sr = ((generator(lr, tz.text_prior(recognizer, lr)) + 1.0) / 2.0).clamp(0, 1)
            out.append((sr.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8))
    return np.concatenate(out) if out else np.zeros((0, HR_SIZE[1], HR_SIZE[0], 3), np.uint8)


def bicubic_words(lr_uint8):
    return np.stack([cv2.resize(w, HR_SIZE, interpolation=cv2.INTER_CUBIC) for w in lr_uint8]) if len(lr_uint8) else lr_uint8


def assemble(line, word_imgs, words, alpha=1.0, max_h=np.inf):
    """The far line with its word boxes replaced by restored words (alpha < 1: blend with the original).
    Only words shorter than max_h px in the far photo are replaced (the size gate); taller ones stay raw."""
    img = read_rgb(REPO / line["input_path"]).astype(np.float32)
    for wi in line["words"]:
        w = words[wi]
        if w["native_h"] >= max_h:
            continue
        box = img[w["y0"]:w["y1"], w["x0"]:w["x1"]]
        new = cv2.resize(word_imgs[wi], (box.shape[1], box.shape[0]), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        img[w["y0"]:w["y1"], w["x0"]:w["x1"]] = alpha * new + (1 - alpha) * box
    return np.clip(img + 0.5, 0, 255).astype(np.uint8)


_PADDLE = None


def paddle_read(images_rgb):
    global _PADDLE
    if _PADDLE is None:
        from paddleocr import TextRecognition
        _PADDLE = TextRecognition(model_name="PP-OCRv6_medium_rec")
    out = []
    for i in range(0, len(images_rgb), 64):
        for p in _PADDLE.predict([cv2.cvtColor(x, cv2.COLOR_RGB2BGR) for x in images_rgb[i:i + 64]]):
            out.append((str(p["rec_text"]), float(p["rec_score"])))
    return out


def lines_cer(lines, word_imgs, words, alpha=1.0, max_h=np.inf):
    imgs = [assemble(ln, word_imgs, words, alpha, max_h) if word_imgs is not None else read_rgb(REPO / ln["input_path"])
            for ln in lines]
    reads = paddle_read(imgs)
    return [line_cer(ln["label"], t) for ln, (t, _) in zip(lines, reads)], reads


def eval_lines(lines, split):
    return [ln for ln in lines if ln["split"] == split and ln["cer_eligible"] == "yes" and ln["label"] and ln["words"]]


def small_word_lines(lines, words, max_h):
    """Lines with at least one word the network is allowed to touch (validation must look like the test board)."""
    return [ln for ln in lines if any(words[wi]["native_h"] < max_h for wi in ln["words"])]


# ----------------------------------------------------------------------------------------------- train
def train(s: Settings) -> None:
    import torch
    tz, generator, recognizer = models(device := tz_device())
    lr_all, hr_all, lines, words = load_prepared()
    saved = torch.load(BASE_CHECKPOINT, map_location=device, weights_only=True)
    generator.load_state_dict(saved["generator"])
    print(f"start: {BASE_CHECKPOINT.relative_to(REPO)} (TextZoom epoch {saved.get('epoch')}) on {device}")
    all_train = [w for w in words if w["split"] == "train"]
    keep = [w for w in all_train if s.min_native_h <= w["native_h"] < s.sr_max_h]
    too_small = sum(w["native_h"] < s.min_native_h for w in all_train)
    print(f"training words: {len(keep)} of {len(all_train)} kept ({s.min_native_h:g} <= box < {s.sr_max_h:g} px in the far photo; "
          f"{too_small} too small to read, {len(all_train) - len(keep) - too_small} too big to need SR)")
    for src in sorted({lines[w["line"]]["source"] for w in keep}):
        print(f"  {src}: {sum(lines[w['line']]['source'] == src for w in keep)} words")
    if not keep and s.smoke:
        keep = all_train
        print("  smoke run: no word in the size range, training on all words just to test the code")
    train_idx = np.array([w["word"] for w in keep])
    val_all = eval_lines(lines, "val")
    val_lines = small_word_lines(val_all, words, s.sr_max_h) or val_all
    assert len(train_idx) and val_lines, "no training words or no validation lines: run `prepare` first"
    textzoom = None
    if s.textzoom_per_batch:
        tzdir = REPO / "data" / "textzoom"
        textzoom = torch.utils.data.ConcatDataset([tz.TextZoomPairs(tzdir / "train1"), tz.TextZoomPairs(tzdir / "train2")])
    loss_cfg = SimpleNamespace(prior_weight=s.prior_weight, gradient_weight=s.gradient_weight)
    opt = torch.optim.Adam(generator.parameters(), lr=s.learning_rate)
    rng = np.random.default_rng(s.seed)
    torch.manual_seed(s.seed)

    val_words = np.array(sorted({wi for ln in val_lines for wi in ln["words"]}))

    def val_cer():
        sr = np.zeros((len(lr_all), HR_SIZE[1], HR_SIZE[0], 3), np.uint8)
        sr[val_words] = restore(tz, generator, recognizer, lr_all[val_words], device)   # only the words read below
        c, _ = lines_cer(val_lines, sr, words, max_h=s.sr_max_h)
        return float(np.mean(c))

    base_raw = float(np.mean(lines_cer(val_lines, None, words)[0]))
    best = dict(cer=val_cer(), epoch=0, state=copy.deepcopy(generator.state_dict()))
    history = [dict(epoch=0, train_loss=np.nan, val_cer=best["cer"])]
    print(f"validation: {len(val_lines)} of {len(val_all)} lines (those with a word < {s.sr_max_h:g} px) | "
          f"raw input CER {base_raw:.3f} | epoch 0 (pretrained) CER {best['cer']:.3f}")
    bad = 0
    for ep in range(1, s.epochs + 1):
        generator.train()
        t0, total = time.time(), []
        for _ in range(s.steps_per_epoch):
            pick = rng.choice(train_idx, s.our_per_batch)
            lr, hr = to_tensor(lr_all[pick]), to_tensor(hr_all[pick])
            if textzoom is not None:
                tzb = [textzoom[int(i)] for i in rng.integers(0, len(textzoom), s.textzoom_per_batch)]
                lr = torch.cat([lr, torch.stack([b[0] for b in tzb])])
                hr = torch.cat([hr, torch.stack([b[1] for b in tzb])])
            lr, hr = lr.to(device), hr.to(device)
            opt.zero_grad(set_to_none=True)
            loss = tz.losses(generator, recognizer, lr, hr, loss_cfg)[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 5.0)
            opt.step()
            total.append(loss.item())
        v = val_cer()
        history.append(dict(epoch=ep, train_loss=float(np.mean(total)), val_cer=v))
        note = ""
        if v < best["cer"] - 1e-4:
            best, bad, note = dict(cer=v, epoch=ep, state=copy.deepcopy(generator.state_dict())), 0, "  * best"
        else:
            bad += 1
        print(f"epoch {ep:>2}: loss {np.mean(total):.4f} | val CER {v:.3f} (raw {base_raw:.3f}) | {time.time() - t0:.0f} s{note}")
        if bad >= s.patience:
            print(f"no better validation CER for {s.patience} epochs: stopping")
            break
    BEST_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"generator": {k: v.cpu() for k, v in best["state"].items()}, "epoch": best["epoch"],
                "val_cer": best["cer"], "val_raw_cer": base_raw, "base": str(BASE_CHECKPOINT.relative_to(REPO)),
                "settings": asdict(s)}, BEST_CHECKPOINT)
    json.dump(history, open(OUT / "training_history.json", "w"), indent=1, default=float)
    print(f"kept epoch {best['epoch']} (val CER {best['cer']:.3f}) -> {BEST_CHECKPOINT.relative_to(REPO)}")
    if best["epoch"] == 0:
        print("NOTE: no epoch beat the pretrained model on validation CER: the saved model IS the pretrained one")


def tz_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ----------------------------------------------------------------------------------------------- prior check
CRNN_CHARS = "-0123456789abcdefghijklmnopqrstuvwxyz"     # index 0 = CTC blank, as in build_recognizer


def crnn_read(tz, recognizer, imgs_uint8, device, bs=128):
    """Greedy CTC reading of the frozen CRNN: the same network and the same input the text prior comes from."""
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs_uint8), bs):
            x = to_tensor(imgs_uint8[i:i + bs]).to(device)
            best = recognizer(tz.recognizer_input(x)).argmax(-1).permute(1, 0).cpu().numpy()   # batch, time
            for seq in best:
                text, prev = [], 0
                for k in seq:
                    if k != prev and k != 0:
                        text.append(CRNN_CHARS[k])
                    prev = k
                out.append("".join(text))
    return out


def word_labels(line):
    """Per-box word labels: only when the line's label has exactly one token per word box (else None).
    Tokens are reduced to the CRNN's alphabet (a-z, 0-9); a token with nothing left (e.g. '&') is skipped."""
    import re
    toks = line["label"].split()
    if len(toks) != len(line["words"]):
        return None
    return [re.sub(r"[^0-9a-z]", "", t.lower()) for t in toks]


def prior_check(s: Settings) -> None:
    """Does the text prior know which letters it is looking at? CRNN CER on the LR words (what the prior sees)
    and on the HR clear words (the best the CRNN can do on this handwriting). No training."""
    import pandas as pd
    tz, _, recognizer = models(device := tz_device())
    lr_all, hr_all, lines, words = load_prepared()
    sets = {"Airdrop B3 (unseen board)": lambda ln: ln["group"] == TEST_GROUP,
            "Airdrop synthetic, validation": lambda ln: ln["source"] == "airdrop_synth" and ln["split"] == "val",
            "17 Oct held-out photos": lambda ln: ln["source"] == "17oct" and ln["split"] == "test"}
    rows = []
    for set_name, keep in sets.items():
        ls = [ln for ln in lines if keep(ln) and ln["label"] and ln["words"]]
        pairs = [(ln, wi, lab) for ln in ls for wi, lab in zip(ln["words"], word_labels(ln) or []) if lab]
        if not pairs:
            continue
        idx = np.array([wi for _, wi, _ in pairs])
        lr_read, hr_read = crnn_read(tz, recognizer, lr_all[idx], device), crnn_read(tz, recognizer, hr_all[idx], device)
        for (ln, wi, lab), a, b in zip(pairs, lr_read, hr_read):
            rows.append(dict(set=set_name, pair_id=ln["pair_id"], word=lab, native_h=words[wi]["native_h"],
                             lr_read=a, hr_read=b, lr_cer=line_cer(lab, a), hr_cer=line_cer(lab, b),
                             lines_in_set=len(ls)))
    if not rows:
        print("prior check: no labelled words found")
        return
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "prior_check_words.csv", index=False)
    out = ["CRNN (text prior source) reading our word crops. Word CER, lower = better; exact = whole word right.",
           "Only lines whose label has one token per word box are used (word count = boxes)."]
    for set_name, part in df.groupby("set", sort=False):
        small = part.native_h < 16
        out += [f"\n=== {set_name}: {len(part)} words from {part.pair_id.nunique()} of {part.lines_in_set.iloc[0]} lines ===",
                f"LR word (what the prior sees):   CER {part.lr_cer.mean():.3f} | exact {(part.lr_cer == 0).mean():.0%}",
                f"HR clear word (CRNN's ceiling):  CER {part.hr_cer.mean():.3f} | exact {(part.hr_cer == 0).mean():.0%}"]
        if small.any() and (~small).any():
            out.append(f"LR CER by size: < 16 px {part[small].lr_cer.mean():.3f} ({small.sum()} words) | "
                       f">= 16 px {part[~small].lr_cer.mean():.3f} ({(~small).sum()} words)")
        ex = part.sort_values("lr_cer", ascending=False).head(5)
        out.append("worst LR readings: " + ", ".join(f"{r.word!r}->{r.lr_read!r}" for r in ex.itertuples()))
    b3 = df[df.set.str.startswith("Airdrop B3")]
    if not b3.empty:
        c = b3.lr_cer.mean()
        verdict = ("mostly RIGHT: the prior points TPGSR at the correct letters" if c < 0.2 else
                   "mostly NOISE: the prior often names the wrong letters; expect TPGSR to draw wrong letters clearly. "
                   "Consider a lower prior influence or fine-tuning the CRNN on our word crops first" if c > 0.4 else
                   "MIXED: right for some words, wrong for others; check the size split above")
        out.append(f"\nB3 verdict: LR CER {c:.3f} -> the prior is {verdict}.")
    text = "\n".join(out)
    print(text)
    (OUT / "prior_check.txt").write_text(text)


# ----------------------------------------------------------------------------------------------- evaluate
def boot(diff_by_photo, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    groups = list(diff_by_photo.values())
    if not groups:
        return np.nan, np.nan
    b = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean() for _ in range(n)]
    return tuple(np.percentile(b, [2.5, 97.5]))


def evaluate(s: Settings) -> None:
    import pandas as pd
    import torch
    tz, generator, recognizer = models(device := tz_device())
    lr_all, hr_all, lines, words = load_prepared()
    generator.load_state_dict(torch.load(BASE_CHECKPOINT, map_location=device, weights_only=True)["generator"])
    sr_pre = restore(tz, generator, recognizer, lr_all, device)
    tuned = torch.load(BEST_CHECKPOINT, map_location=device, weights_only=True)
    generator.load_state_dict(tuned["generator"])
    sr_ft = restore(tz, generator, recognizer, lr_all, device)
    path16 = bicubic_words(lr_all)                     # same 16 px route, no network
    print(f"fine-tuned model: epoch {tuned['epoch']} (val CER {tuned['val_cer']:.3f})")

    # choose size gate, blend weight and confidence margin on VALIDATION only (small-word lines, like B3)
    val_all = eval_lines(lines, "val")
    val = small_word_lines(val_all, words, s.sr_max_h) or val_all
    raw_c, raw_r = lines_cer(val, None, words)
    gh, best_hc = None, None
    for h in s.size_gate_choices:                       # ascending: on a tie the smaller gate (fewer words touched) wins
        c = float(np.mean(lines_cer(val, sr_ft, words, max_h=h)[0]))
        print(f"  validation: size gate, restore words < {h:g} px: CER {c:.3f} (raw {np.mean(raw_c):.3f})")
        if best_hc is None or c < best_hc - 1e-4:
            gh, best_hc = h, c
    best_alpha, best_c = 1.0, None
    for a in (0.3, 0.5, 0.7, 1.0):
        c = float(np.mean(lines_cer(val, sr_ft, words, alpha=a, max_h=gh)[0]))
        print(f"  validation: blend alpha {a}: CER {c:.3f} (raw {np.mean(raw_c):.3f})")
        if best_c is None or c < best_c - 1e-4:
            best_alpha, best_c = a, c
    ft_c, ft_r = lines_cer(val, sr_ft, words, alpha=best_alpha, max_h=gh)
    best_m, best_gc = 0.0, None
    for m in (0.0, 0.02, 0.05, 0.1):
        g = [fc if fr[1] > rr[1] + m else rc for fc, fr, rc, rr in zip(ft_c, ft_r, raw_c, raw_r)]
        if best_gc is None or np.mean(g) < best_gc - 1e-4:
            best_m, best_gc = m, float(np.mean(g))
    candidates = {"TPGSR fine-tuned": best_hc, f"blend (alpha {best_alpha})": best_c, f"gate (margin {best_m})": best_gc}
    final = min(candidates, key=candidates.get)
    print(f"chosen on validation: size gate {gh:g} px, blend alpha {best_alpha}, gate margin {best_m}; final version = {final} "
          f"(validation CER {candidates[final]:.3f}, raw {np.mean(raw_c):.3f})")

    rows = []
    sets = {"Airdrop B3 (unseen board, main test)": [ln for ln in eval_lines(lines, "test") if ln["group"] == TEST_GROUP],
            "17 Oct held-out photos": [ln for ln in lines if ln["source"] == "17oct" and ln["split"] == "test"
                                       and ln["cer_eligible"] == "yes" and ln["label"] and ln["words"]]}
    for set_name, ls in sets.items():
        if not ls:
            continue
        res = {"raw input": lines_cer(ls, None, words), "16px path, no network": lines_cer(ls, path16, words, max_h=gh),
               "TPGSR pretrained": lines_cer(ls, sr_pre, words, max_h=gh), "TPGSR fine-tuned": lines_cer(ls, sr_ft, words, max_h=gh),
               "TPGSR fine-tuned, every word": lines_cer(ls, sr_ft, words),
               f"blend (alpha {best_alpha})": lines_cer(ls, sr_ft, words, alpha=best_alpha, max_h=gh)}
        bl_c, bl_r = res[f"blend (alpha {best_alpha})"]
        rc, rr = res["raw input"]
        res[f"gate (margin {best_m})"] = ([b if br[1] > r_[1] + best_m else r for b, br, r, r_ in zip(bl_c, bl_r, rc, rr)],
                                          [br if br[1] > r_[1] + best_m else r_ for br, r_ in zip(bl_r, rr)])
        for method, (c, r) in res.items():
            for ln, ci, (txt, conf) in zip(ls, c, r):
                rows.append(dict(set=set_name, method=method, pair_id=ln["pair_id"], photo=ln["pair_id"].split("__")[0],
                                 label=ln["label"], read=txt, confidence=conf, cer=ci))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ocr_comparison.csv", index=False)
    lines_out = [f"Size gate chosen on validation: only words < {gh:g} px (far photo) are restored, in every version except "
                 f"'raw input' and 'TPGSR fine-tuned, every word'. The 16px path uses the same gate, so it isolates the network."]
    for set_name, part in df.groupby("set", sort=False):
        raw = part[part.method == "raw input"].set_index("pair_id").cer
        lines_out.append(f"\n=== {set_name}: {raw.size} lines ===")
        photo_of = part[part.method == "raw input"].set_index("pair_id").photo
        for method, m in part.groupby("method", sort=False):
            c = m.set_index("pair_id").cer
            d = (raw - c.reindex(raw.index))
            by_photo = {p: d[(photo_of == p).values].values for p in photo_of.unique()}
            lo, hi = boot(by_photo)
            lines_out.append(f"{method:<26} CER {c.mean():.3f} | vs raw {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | "
                             f"better {(d > 1e-9).sum()} / same {(d.abs() <= 1e-9).sum()} / worse {(d < -1e-9).sum()}")
    main = df[df.set.str.startswith("Airdrop B3")]
    if not main.empty:
        mean = main.groupby("method").cer.mean()
        raw_i = main[main.method == "raw input"].set_index("pair_id").cer
        d = raw_i - main[main.method == final].set_index("pair_id").cer.reindex(raw_i.index)
        ok = (mean[final] < mean["raw input"] and mean[final] < mean["16px path, no network"]
              and (d > 1e-9).sum() > (d < -1e-9).sum())
        lines_out.append(f"\nVERDICT (rule fixed before training, unseen board; version chosen on validation = {final}):"
                         f"\n  CER {mean[final]:.3f} vs raw input {mean['raw input']:.3f} and 16px path {mean['16px path, no network']:.3f};"
                         f" {(d > 1e-9).sum()} lines better / {(d < -1e-9).sum()} worse -> {'PASSES' if ok else 'FAILS'}")
    text = "\n".join(lines_out)
    print(text)
    (OUT / "summary.txt").write_text(text)
    visual(main, lines, words, sr_ft, gh)


def visual(main, lines, words, sr_ft, max_h, n=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if main.empty:
        return
    by_id = {ln["pair_id"]: ln for ln in lines}
    raw = main[main.method == "raw input"].set_index("pair_id")
    ft = main[main.method == "TPGSR fine-tuned"].set_index("pair_id")
    diff = (raw.cer - ft.cer).sort_values()
    pick = list(diff.index[:n // 2]) + list(diff.index[-(n - n // 2):])
    fig, axes = plt.subplots(len(pick) * 3, 1, figsize=(12, 2.1 * 3 * len(pick)))
    for k, pid in enumerate(pick):
        ln = by_id[pid]
        imgs = [(read_rgb(REPO / ln["target_path"]), "clear photo (target)", ""),
                (read_rgb(REPO / ln["input_path"]), "raw input (far)", raw.loc[pid, "read"]),
                (assemble(ln, sr_ft, words, max_h=max_h), "TPGSR fine-tuned", ft.loc[pid, "read"])]
        for j, (im, title, txt) in enumerate(imgs):
            a = axes[3 * k + j]
            a.imshow(im); a.axis("off")
            a.set_title(f"{title}" + (f" | PaddleOCR: {txt!r}" if txt else f" | truth: {ln['label']!r}"), fontsize=8, loc="left")
    plt.tight_layout()
    fig.savefig(OUT / "visual_comparison.png", dpi=120)
    plt.close(fig)
    print(f"saved {(OUT / 'visual_comparison.png').relative_to(REPO)} (2 lines that got worse, 2 that got better)")


# ----------------------------------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["prepare", "prior", "train", "evaluate", "all"])
    p.add_argument("--smoke", action="store_true", help="tiny run to check that everything works")
    p.add_argument("--epochs", type=int)
    p.add_argument("--steps-per-epoch", type=int)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--textzoom-per-batch", type=int)
    a = p.parse_args()
    s = Settings(smoke=a.smoke)
    if a.smoke:
        s.epochs, s.steps_per_epoch, s.our_per_batch, s.textzoom_per_batch, s.patience = 2, 3, 4, 4, 2
    for k in ("epochs", "steps_per_epoch", "learning_rate", "textzoom_per_batch"):
        if getattr(a, k) is not None:
            setattr(s, k, getattr(a, k))
    if s.smoke:
        global OUT, BEST_CHECKPOINT
        OUT = OUT.parent / "tpgsr_whiteboard_smoke"
        BEST_CHECKPOINT = OUT / "tpgsr_whiteboard_smoke.pt"     # never overwrites a real model
    random.seed(s.seed)
    np.random.seed(s.seed)
    if a.step in ("prepare", "all"):
        prepare(s)
    if a.step in ("prior", "all"):
        prior_check(s)
    if a.step in ("train", "all"):
        train(s)
    if a.step in ("evaluate", "all"):
        evaluate(s)


if __name__ == "__main__":
    main()
