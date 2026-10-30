#!/usr/bin/env python3
"""Train the frozen line recogniser used by the M2 recognition loss (see line_recognizer.py).

Training data
  * IAM handwritten lines (Assignment/iam_line_paddle, train split): 6,482 lines, pen on paper.
  * Our own labelled whiteboard line crops, so the recogniser knows marker on a board:
      - Airdrop crops (airdrop_rec_paddle, accepted + duplicate) of boards B1, B4, B5 only. B2 (the
        SR validation board) and B3 (the SR test board, CNN) are left out, so the recogniser never
        sees the handwriting it will later help to judge.
      - 17 Oct crops (17oct_rec_paddle, accepted) except those from the SR validation and test photos.
    They are repeated OWN_REPEAT times per epoch so the whiteboard is not drowned by IAM.
  Mild augmentation only (blur, slight resolution loss, ink strength, stroke width, shear): the
  recogniser should read CLEAR lines well. As a loss it then pushes the SR output towards clear
  strokes; a recogniser that also reads blur well would push much less.

Checkpoint choice: lowest CER on IAM validation lines + 17 Oct crops of the SR VALIDATION photos.
The CNN board (B3) crops are only REPORTED (never used to choose), as a whiteboard sanity check.

Run from whiteboard-ocr/:  python scripts/train_line_recognizer.py [--smoke] [--self-test]
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import line_recognizer as R  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ASSIGNMENT = REPO.parent
IAM = ASSIGNMENT / "iam_line_paddle"
AIRDROP = ASSIGNMENT / "airdrop_rec_paddle"
OCT17 = ASSIGNMENT / "17oct_rec_paddle"
DEFAULT_OUT = REPO / "models" / "superres" / "line_recognizer_crnn.pt"
EXCLUDED_BOARDS = {"B2", "B3"}                     # SR validation (television) and test (CNN) boards
OCT17_VAL_PHOTOS = {"IMG_4736", "IMG_4705", "IMG_4673"}   # pair_data.VAL_INPUT_PHOTOS
OCT17_TEST_PHOTOS = {"IMG_4748", "IMG_4750", "IMG_4697"}  # pair_data.TEST_INPUT_PHOTOS
OWN_REPEAT = 4


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--iam-val-lines", type=int, default=400, help="IAM validation lines used per check (speed)")
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--smoke", action="store_true", help="1 short epoch on a small subset: check that it runs")
    ap.add_argument("--self-test", action="store_true", help="check the model and CTC pieces and exit")
    return ap.parse_args()


# ------------------------------------------------------------------------------------------------
def read_grey(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def iam_rows(split: str):
    rows = []
    with (IAM / f"{split}.txt").open(encoding="utf-8") as fh:
        for line in fh:
            path, _, text = line.rstrip("\n").partition("\t")
            if text:
                rows.append({"path": IAM / path, "text": text, "source": f"IAM {split}"})
    return rows


def own_rows():
    """(train rows, choose rows, report rows) from our labelled whiteboard crops."""
    train, choose, report = [], [], []
    with (AIRDROP / "manifest.csv").open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["status"] not in ("accepted", "duplicate") or not r["label"].strip():
                continue
            row = {"path": AIRDROP / r["crop"], "text": r["label"], "source": f"Airdrop {r['board']}"}
            if r["board"] == "B3":
                report.append(row)
            elif r["board"] not in EXCLUDED_BOARDS:
                train.append(row)
    with (OCT17 / "manifest.csv").open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["status"] != "accepted" or not r["label"].strip():
                continue
            row = {"path": OCT17 / r["crop"], "text": r["label"], "source": "17 Oct"}
            if r["photo"] in OCT17_VAL_PHOTOS:
                choose.append(row)
            elif r["photo"] not in OCT17_TEST_PHOTOS:
                train.append(row)
    return train, choose, report


def prepare(rows):
    """Load every crop once and keep its INK image (uint8, HEIGHT tall) in memory."""
    out = []
    for r in rows:
        rgb = read_grey(r["path"])
        if rgb is None or min(rgb.shape[:2]) < 8:
            continue
        ink = R.ink_from_rgb(rgb)
        out.append({**r, "ink": np.rint(ink * 255).astype(np.uint8)})
    return out


def augment(ink_u8: np.ndarray, rng: random.Random) -> np.ndarray:
    """Mild, recogniser-style augmentation on an INK image (ink = bright)."""
    ink = ink_u8.astype(np.float32) / 255.0
    h, w = ink.shape
    if rng.random() < 0.5:                                       # shear (slanted handwriting)
        s = rng.uniform(-0.25, 0.25)
        m = np.float32([[1, s, -s * h / 2], [0, 1, 0]])
        ink = cv2.warpAffine(ink, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    if rng.random() < 0.3:                                       # stroke width
        k = np.ones((2, 2), np.uint8)
        ink = cv2.dilate(ink, k) if rng.random() < 0.5 else cv2.erode(ink, k)
    if rng.random() < 0.4:                                       # slight resolution loss
        f = rng.uniform(1.0, 1.8)
        small = cv2.resize(ink, (max(8, int(w / f)), max(8, int(h / f))), interpolation=cv2.INTER_AREA)
        ink = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.4:                                       # slight blur
        ink = cv2.GaussianBlur(ink, (0, 0), rng.uniform(0.3, 1.0))
    ink = ink * rng.uniform(0.7, 1.15)                           # ink strength
    if rng.random() < 0.5:                                       # sensor noise
        noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0.0, 0.03, ink.shape)
        ink = ink + noise.astype(np.float32)
    return np.clip(ink, 0.0, 1.0)


def evaluate(model, charset, items, device, batch_size=32):
    model.eval()
    errors, preds = [], []
    for i in range(0, len(items), batch_size):
        part = items[i:i + batch_size]
        batch, lengths = R.pad_batch([p["ink"].astype(np.float32) / 255.0 for p in part])
        texts = R.read(model, charset, batch.to(device), lengths)
        for p, t in zip(part, texts):
            errors.append(R.cer(p["text"], t))
            preds.append(t)
    return float(np.mean(errors)) if errors else float("nan"), preds


# ------------------------------------------------------------------------------------------------
def self_test() -> None:
    torch.manual_seed(0)
    cs = R.Charset("abc")              # sorted: a=1, b=2, c=3 (0 is the CTC blank)
    checks = [("encode/decode", cs.decode_greedy([1, 1, 0, 2, 2, 0, 0, 1]) == "aba"),
              ("unknown chars dropped", cs.encode("abz") == [1, 2]),
              ("CER", abs(R.cer("Hello world", "hello  wor1d") - 1 / 11) < 1e-9)]
    model = R.LineCRNN(len(cs))
    batch, lengths = R.pad_batch([np.random.rand(R.HEIGHT, 120).astype(np.float32),
                                  np.random.rand(R.HEIGHT, 64).astype(np.float32)])
    log_probs = model(batch)
    checks.append(("output shape (T, B, C)", tuple(log_probs.shape) == (batch.shape[3] // R.DOWNSAMPLE, 2, len(cs))))
    x = torch.rand(1, 3, 95, 400, requires_grad=True)
    ink = R.ink_from_sr_output(x)
    loss = R.ctc_loss(model(ink), torch.tensor([ink.shape[3] // R.DOWNSAMPLE]), [cs.encode("abc ab")])
    loss.backward()
    checks.append(("CTC finite, gradient reaches the image", bool(torch.isfinite(loss)) and float(x.grad.abs().sum()) > 0))
    for h in (95, 300):                                  # no pre-shrink, and a 2x2 pre-shrink
        grey = 1.0 - cv2.GaussianBlur((np.random.rand(h, 4 * h) > 0.97).astype(np.float32), (0, 0), 2.0) * 8
        grey = np.clip(grey, 0, 1).astype(np.float32)
        a = R.ink_from_norm_grey(grey)
        b = R.ink_from_sr_output(torch.from_numpy(grey)[None, None].repeat(1, 3, 1, 1))[0, 0].numpy()
        checks.append((f"numpy and torch preprocessing agree (height {h})",
                       a.shape == b.shape and float(np.abs(a - b).mean()) < 0.01))
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not all(ok for _, ok in checks):
        raise SystemExit("self-test FAILED")
    print("recogniser self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    own_train, own_choose, own_report = own_rows()
    iam_train, iam_val = iam_rows("train"), iam_rows("val")
    rng = random.Random(args.seed)
    iam_val = rng.sample(iam_val, min(args.iam_val_lines, len(iam_val)))
    if args.smoke:
        iam_train, iam_val = iam_train[:200], iam_val[:32]
        own_train, own_choose, own_report = own_train[:40], own_choose[:16], own_report[:16]
    t0 = time.time()
    print("loading and normalising crops (once) ...")
    sets = {name: prepare(rows) for name, rows in
            [("iam_train", iam_train), ("own_train", own_train), ("iam_val", iam_val),
             ("own_choose", own_choose), ("own_report", own_report)]}
    print(f"  done in {time.time() - t0:.0f}s: " + ", ".join(f"{k} {len(v)}" for k, v in sets.items()))

    charset = R.Charset("".join(p["text"] for p in sets["iam_train"] + sets["own_train"]))
    dropped = sum(len(p["text"]) - len(charset.encode(p["text"])) for p in sets["own_choose"] + sets["own_report"])
    print(f"charset: {len(charset.chars)} characters; characters of held-out labels not in it: {dropped}")
    train_items = sets["iam_train"] + sets["own_train"] * OWN_REPEAT
    choose_items = sets["iam_val"] + sets["own_choose"]

    model = R.LineCRNN(len(charset)).to(device)
    print(f"device: {device} | parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")
    steps_per_epoch = math.ceil(len(train_items) / args.batch_size) if not args.smoke else 10
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate,
                                                    total_steps=args.epochs * steps_per_epoch, pct_start=0.15)
    best, best_epoch, bad = math.inf, 0, 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train_items)))
        rng.shuffle(order)
        losses = []
        for step in range(steps_per_epoch):
            idx = order[step * args.batch_size:(step + 1) * args.batch_size]
            if not idx:
                break
            part = [train_items[i] for i in idx]
            batch, lengths = R.pad_batch([augment(p["ink"], rng) for p in part])
            labels = [charset.encode(p["text"]) for p in part]
            keep = [k for k, lab in enumerate(labels) if lab]
            if not keep:
                continue
            log_probs = model(batch.to(device))
            loss = R.ctc_loss(log_probs[:, keep], lengths[keep], [labels[k] for k in keep])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss))
        iam_cer, _ = evaluate(model, charset, sets["iam_val"], device)
        own_cer, _ = evaluate(model, charset, sets["own_choose"], device)
        b3_cer, _ = evaluate(model, charset, sets["own_report"], device)
        score = float(np.mean([iam_cer, own_cer]))
        improved = score < best - 1e-4
        if improved:
            best, best_epoch, bad = score, epoch, 0
            R.save(args.output, model, charset, epoch=epoch, iam_val_cer=iam_cer, oct17_val_cer=own_cer,
                   airdrop_b3_cer=b3_cer, train_sources={"IAM lines": len(sets["iam_train"]),
                                                         "own whiteboard crops": len(sets["own_train"])})
        else:
            bad += 1
        print(f"epoch {epoch:02d}: CTC loss {np.mean(losses):.3f} | CER IAM val {iam_cer:.3f}, 17 Oct val photos "
              f"{own_cer:.3f} | (report only) CNN board B3 crops {b3_cer:.3f}{' * best' if improved else ''}")
        if bad >= args.patience:
            print(f"early stopping after {args.patience} epochs without improvement")
            break
    model, charset, info = R.load(args.output, device)
    _, preds = evaluate(model, charset, sets["own_report"][:8], device)
    print(f"saved epoch {best_epoch} to {args.output}")
    print("examples on the CNN board (never trained on):")
    for p, t in zip(sets["own_report"][:8], preds):
        print(f"  {p['text']!r:45} -> {t!r}")


if __name__ == "__main__":
    main()
