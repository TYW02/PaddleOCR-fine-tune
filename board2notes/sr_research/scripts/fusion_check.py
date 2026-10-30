#!/usr/bin/env python3
"""Does reading the same line in SEVERAL photos beat reading it in one? (multi-photo reading fusion, no training)

Why: super-resolution reinterprets ONE photo; a second or third photo adds new measurements of the writing.
Here each photo's own crop of a line is read by the app's reader (PP-OCRv6_medium_rec, pretrained) and the
readings are fused at the TEXT level (scripts/reading_fusion.py: no pixel alignment needed):
  hc      the reading PaddleOCR is most confident about
  mv      the most frequent reading
  medoid  the reading closest to all others
  vote    character vote after aligning every reading to the medoid (ROVER)
  oracle  the best photo of the subset (needs the truth: an upper bound, not a usable rule)

Crops: every labelled line crop of the reader fine-tune lists (outputs/reader_finetune/data/*_meta.csv:
train_own, val, test_b3, test_17oct; 573 crops of boards B1-B5 and 17 Oct). The pretrained reader never trained
on any of them, so all boards are fair here. A "line" = the same text on the same board; one crop per photo.

DECISION RULE (fixed before reading):
  * Scenario: the user takes K = 3 photos of the board. For every line with at least 3 photos, random 3-photo
    subsets are fused (up to 40 per line, seeded) and averaged; "single" = the expected CER of ONE photo.
  * The rule (hc / mv / medoid / vote) is CHOSEN on all boards except B3 (lowest mean CER).
  * B3 decides: fusion helps if the chosen rule's mean CER on B3 is below "single" by more than 0.02 (the reader
    noise between two photos of the same text) AND more B3 lines get better than worse.
Reported as well: 2 photos per line, every board separately, and lines whose photos all show text under 16 px.
Assumption to state in the report: which crops show the same line is known (here from the labels). In the app
the lines of different photos must be matched (box overlap after perspective correction, or text similarity).

Run on the Mac from the Assignment folder:
  .venv-paddle/bin/python whiteboard-ocr/scripts/fusion_check.py
Output: whiteboard-ocr/outputs/fusion_check/readings.csv, lines.csv, summary.txt
"""
from __future__ import annotations

import argparse
import csv
import itertools
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reading_fusion as F  # noqa: E402
from finetune_reader import ASSIGNMENT, REPO  # noqa: E402

DATA = REPO / "outputs" / "reader_finetune" / "data"
OUT = REPO / "outputs" / "fusion_check"
FILES = ("train_own", "val", "test_b3", "test_17oct")
TEST_SOURCE = "airdrop B3"
K = 3
NOISE = 0.02
SMALL_PX = 16
SUBSETS = 40
SEED = 0


def load_crops():
    rows, seen = [], set()
    for f in FILES:
        for r in csv.DictReader((DATA / f"{f}_meta.csv").open(encoding="utf-8")):
            if r["path"] not in seen:
                seen.add(r["path"])
                rows.append(dict(file=f, source=r["source"], photo=r["photo"], path=r["path"], label=r["label"]))
    missing = [r["path"] for r in rows if not (ASSIGNMENT / r["path"]).exists()]
    if missing:
        sys.exit(f"{len(missing)} crops missing, e.g. {missing[:2]}")
    return rows


def read(rows, bs=32):
    import cv2
    from capture_check import text_height_px
    from paddleocr import TextRecognition
    model = TextRecognition(model_name="PP-OCRv6_medium_rec")
    t0 = time.time()
    for i in range(0, len(rows), bs):
        part = rows[i:i + bs]
        imgs = [cv2.imread(str(ASSIGNMENT / r["path"])) for r in part]
        for r, img, res in zip(part, imgs, model.predict(imgs)):
            r["read"], r["conf"] = str(res["rec_text"]), round(float(res["rec_score"]), 4)
            r["cer"] = round(F.cer(r["label"], r["read"]), 4)
            r["text_height_px"] = round(float(text_height_px(img)), 1)
        print(f"  read {min(i + bs, len(rows))}/{len(rows)} crops", end="\r", flush=True)
    print(f"\nread {len(rows)} crops in {time.time() - t0:.0f} s")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "readings.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def line_groups(rows, max_height=None):
    """{(source, text): (truth, [(read, conf), ...])}, one crop per photo; optionally only small-text crops."""
    groups, photos = defaultdict(list), defaultdict(set)
    for r in rows:
        h = float(r["text_height_px"])
        if max_height is not None and not h < max_height:     # NaN heights are left out too
            continue
        key = (r["source"], F.normalise(r["label"]))
        if r["photo"] in photos[key]:
            continue
        photos[key].add(r["photo"])
        groups[key].append((r["read"], float(r["conf"])))
    return {k: (k[1], v) for k, v in groups.items()}


def per_line(groups, k, rng):
    """For every line with >= k photos: expected one-photo CER, and each rule (and the oracle) averaged over
    random k-photo subsets."""
    out = {}
    for key, (truth, readings) in sorted(groups.items()):
        n = len(readings)
        if n < max(k, 2):
            continue
        combos = list(itertools.combinations(range(n), k))
        if len(combos) > SUBSETS:
            combos = rng.sample(combos, SUBSETS)
        line = {"single": sum(F.cer(truth, t) for t, _ in readings) / n, "photos": n}
        for rule in F.RULES + ("oracle",):
            scores = []
            for combo in combos:
                subset = [readings[i] for i in combo]
                scores.append(min(F.cer(truth, t) for t, _ in subset) if rule == "oracle"
                              else F.cer(truth, F.fuse(subset, rule)))
            line[rule] = sum(scores) / len(scores)
        out[key] = line
    return out


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def table(lines, title):
    cols = ("single",) + F.RULES + ("oracle",)
    text = [f"{title}: {len(lines)} lines",
            "    " + "  ".join(f"{c:>7}" for c in cols)]
    if lines:
        text.append("    " + "  ".join(f"{mean(l[c] for l in lines.values()):7.3f}" for c in cols))
    return text


def analyse():
    rows = list(csv.DictReader((OUT / "readings.csv").open(encoding="utf-8")))
    rng = random.Random(SEED)
    text = [f"Multi-photo reading fusion with PP-OCRv6 (mean line CER, lower = better; {len(rows)} crops).",
            "single = expected CER of ONE photo (today's pipeline); hc / mv / medoid / vote = fused readings;",
            "oracle = best photo of the subset (needs the truth: an upper bound).", ""]
    groups = line_groups(rows)
    main = per_line(groups, K, rng)
    sel = {k: v for k, v in main.items() if k[0] != TEST_SOURCE}
    test = {k: v for k, v in main.items() if k[0] == TEST_SOURCE}
    text += table(sel, f"{K} photos per line, boards used to CHOOSE the rule (all but B3)")
    chosen = min(F.RULES, key=lambda r: mean(l[r] for l in sel.values()))
    text.append(f"    -> rule chosen on these boards: {chosen}")
    text.append("")
    text += table(test, f"{K} photos per line, B3 (decides)")
    d = [l["single"] - l[chosen] for l in test.values()]                 # + = fusion better
    better, worse = sum(x > 1e-9 for x in d), sum(x < -1e-9 for x in d)
    s_cer, f_cer = mean(l["single"] for l in test.values()), mean(l[chosen] for l in test.values())
    ok = bool(test) and f_cer < s_cer - NOISE and better > worse
    text.append(f"    B3 lines better / same / worse with '{chosen}': {better} / {len(d) - better - worse} / {worse}")
    text.append(f"\nDECISION (rule fixed before reading): B3 single {s_cer:.3f} -> fused ({chosen}) {f_cer:.3f} "
                f"({f_cer - s_cer:+.3f}; needs < -{NOISE} and more lines better than worse) -> "
                + ("FUSION HELPS" if ok else "NO CLEAR GAIN"))

    text.append("\nFor information:")
    two = per_line(groups, 2, random.Random(SEED))
    text += ["  " + t for t in table({k: v for k, v in two.items() if k[0] == TEST_SOURCE}, "2 photos per line, B3")]
    text += ["  " + t for t in table({k: v for k, v in two.items() if k[0] != TEST_SOURCE}, "2 photos per line, other boards")]
    small = per_line(line_groups(rows, SMALL_PX), K, random.Random(SEED))
    text += ["  " + t for t in table(small, f"{K} photos per line, only photos with text under {SMALL_PX} px (all boards)")]
    for source in sorted({k[0] for k in main}):
        text += ["  " + t for t in table({k: v for k, v in main.items() if k[0] == source}, f"{K} photos per line, {source}")]

    text.append(f"\nB3 lines (truth | single -> {chosen} | photos):")
    for (source, truth), l in test.items():
        text.append(f"  {truth!r}: {l['single']:.3f} -> {l[chosen]:.3f} | {l['photos']} photos")
    with (OUT / "lines.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["board", "text", "photos", "single", *F.RULES, "oracle"])
        for (source, truth), l in main.items():
            w.writerow([source, truth, l["photos"], *(round(l[c], 4) for c in ("single",) + F.RULES + ("oracle",))])
    s = "\n".join(text)
    print(s)
    (OUT / "summary.txt").write_text(s + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analyse-only", action="store_true", help="reuse outputs/fusion_check/readings.csv")
    a = ap.parse_args()
    F.self_test()
    if not a.analyse_only:
        rows = load_crops()
        print(f"{len(rows)} labelled line crops; reading them with PP-OCRv6 ...")
        read(rows)
    analyse()


if __name__ == "__main__":
    main()
