"""Fuse PaddleOCR readings of the SAME text line from several photos (multi-frame, at the text level).

Why: one far photo can lose a stroke that another photo of the same line kept. On the CNN board, picking
the best photo per line would lower CER from 0.157 to 0.094, so the information is often present in SOME
photo. Fusing readings recovers it without inventing strokes (unlike generative / recognition-guided SR).

Rules, following Nascimento et al. 2025 ("Toward Advancing License Plate Super-Resolution in Real-World
Scenarios"), who fused OCR outputs of 5 frames per plate:
  hc      highest confidence: keep the reading PaddleOCR is most sure of
  mv      majority vote: the most frequent reading (ties -> higher total confidence)
  medoid  the reading closest (edit distance) to all the others
  vote    character-position vote. Plates have a fixed length; handwritten lines do not, so every reading is
          first ALIGNED to a pivot reading (the medoid) by edit distance, then each position (and each gap
          between positions, for inserted characters) is decided by a confidence-weighted vote. This is
          the classic ROVER idea from speech recognition.
  oracle  the best single photo per line (needs the ground truth: an upper bound, not a usable rule)

Texts are compared in the same normalised form as the notebook's CER (lower case, single spaces).
"""
from __future__ import annotations

import itertools
import random
from collections import defaultdict


def normalise(text: str) -> str:
    return " ".join(str(text).lower().split())


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(reference: str, prediction: str) -> float:
    reference, prediction = normalise(reference), normalise(prediction)
    if not reference:
        return 0.0 if not prediction else 1.0
    return min(1.0, edit_distance(reference, prediction) / len(reference))


def align_to_pivot(pivot: str, hyp: str):
    """Edit-distance alignment of hyp onto pivot. Returns (per_pos, gaps): per_pos[i] is the character of
    hyp aligned to pivot[i] ('' if deleted); gaps[i] is the text hyp inserts BEFORE pivot position i
    (gaps has len(pivot) + 1 entries; the last one is after the end)."""
    n, m = len(pivot), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (pivot[i - 1] != hyp[j - 1]))
    per_pos, gaps = [""] * n, [""] * (n + 1)
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (pivot[i - 1] != hyp[j - 1]):
            per_pos[i - 1] = hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            per_pos[i - 1] = ""                      # hyp skips this pivot character
            i -= 1
        else:
            gaps[i] = hyp[j - 1] + gaps[i]           # hyp has an extra character before pivot[i]
            j -= 1
    return per_pos, gaps


def fuse(readings: list[tuple[str, float]], rule: str) -> str:
    """readings: [(text, confidence)] of one line from several photos -> fused text (normalised)."""
    texts = [normalise(t) for t, _ in readings]
    confs = [max(float(c), 1e-3) for _, c in readings]
    if len(texts) == 1:
        return texts[0]
    if rule == "hc":
        return texts[max(range(len(texts)), key=lambda k: confs[k])]
    if rule == "mv":
        weight = defaultdict(float)
        count = defaultdict(int)
        for t, c in zip(texts, confs):
            count[t] += 1
            weight[t] += c
        return max(count, key=lambda t: (count[t], weight[t]))
    medoid_index = min(range(len(texts)), key=lambda k: (sum(edit_distance(texts[k], u) for u in texts), -confs[k]))
    pivot = texts[medoid_index]
    if rule == "medoid":
        return pivot
    if rule == "vote":
        total = sum(confs)
        pos_votes = [defaultdict(float) for _ in pivot]
        gap_votes = [defaultdict(float) for _ in range(len(pivot) + 1)]
        for t, c in zip(texts, confs):
            per_pos, gaps = align_to_pivot(pivot, t)
            for i, ch in enumerate(per_pos):
                pos_votes[i][ch] += c
            for i, g in enumerate(gaps):
                if g:
                    gap_votes[i][g] += c
        out = []
        for i in range(len(pivot) + 1):
            if gap_votes[i]:
                best_gap, w = max(gap_votes[i].items(), key=lambda kv: kv[1])
                if w > total / 2:                      # insert only if most of the weight agrees
                    out.append(best_gap)
            if i < len(pivot):
                votes = pos_votes[i]
                best = max(votes.items(), key=lambda kv: (kv[1], kv[0] == pivot[i]))[0]
                out.append(best)
        return normalise("".join(out))
    raise ValueError(rule)


RULES = ("hc", "mv", "medoid", "vote")


def evaluate(groups: dict, ks=(2, 3, "all"), max_subsets: int = 40, seed: int = 0) -> list[dict]:
    """groups: {line_key: (ground_truth, [(text, confidence), ...])}.

    For each k, ONLY the lines with at least k photos (k = "all": at least 2) are used, and every rule is
    scored on exactly those lines, so the numbers in one row are comparable:
      single  the expected CER of ONE photo of the line (mean over its photos): today's pipeline
      rules   fused readings of (up to max_subsets) random k-photo subsets, averaged per line
      oracle  the best single photo in each subset (needs the truth: an upper bound)
    Each line counts once. Returns one dict per (k, rule)."""
    rng = random.Random(seed)
    results = []
    for k in ks:
        per_rule = defaultdict(list)
        for key, (truth, readings) in groups.items():
            n = len(readings)
            kk = n if k == "all" else k
            if n < 2 or kk > n:
                continue
            per_rule["single"].append(sum(cer(truth, t) for t, _ in readings) / n)
            combos = list(itertools.combinations(range(n), kk))
            if len(combos) > max_subsets:
                combos = rng.sample(combos, max_subsets)
            for rule in RULES + ("oracle",):
                scores = []
                for combo in combos:
                    subset = [readings[i] for i in combo]
                    if rule == "oracle":
                        scores.append(min(cer(truth, t) for t, _ in subset))
                    else:
                        scores.append(cer(truth, fuse(subset, rule)))
                per_rule[rule].append(sum(scores) / len(scores))
        for rule, values in per_rule.items():
            results.append({"photos per line": k, "rule": rule, "lines": len(values),
                            "mean CER": round(sum(values) / len(values), 4)})
    return results


def self_test() -> None:
    checks = []
    per_pos, gaps = align_to_pivot("cnn achitecture", "cnn architecture")
    checks.append(("alignment finds the inserted 'r'", "".join(gaps) == "r" and "".join(per_pos) == "cnn achitecture"))
    per_pos, gaps = align_to_pivot("visualising", "visalising")
    checks.append(("alignment finds the deleted 'u'", per_pos.count("") == 1 and not "".join(gaps)))
    three = [("CMN Achitecture", 0.8), ("CNN Achitecture", 0.9), ("CNN Achitecture", 0.85)]
    checks.append(("vote fixes CMN by majority", fuse(three, "vote") == "cnn achitecture"))
    checks.append(("hc picks the most confident", fuse(three, "hc") == "cnn achitecture"))
    checks.append(("mv picks the most frequent", fuse(three, "mv") == "cnn achitecture"))
    mixed = [("Visualising cuN", 0.7), ("Visualising CNN", 0.9), ("Visulising CNN", 0.8)]
    checks.append(("vote combines the right parts of different photos", fuse(mixed, "vote") == "visualising cnn"))
    checks.append(("single reading is returned unchanged", fuse([("Hello  World", 0.5)], "vote") == "hello world"))
    groups = {"a": ("CNN Architecture", three), "b": ("Visualising CNN", mixed)}
    res = {(r["photos per line"], r["rule"]): r["mean CER"] for r in evaluate(groups)}
    checks.append(("fused CER <= single-photo CER here", res[("all", "vote")] <= res[("all", "single")]))
    checks.append(("single and fused rows use the same lines",
                   len({r["lines"] for r in evaluate(groups) if r["photos per line"] == "all"}) == 1))
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not all(ok for _, ok in checks):
        raise SystemExit("reading_fusion self-test FAILED")
    print("reading_fusion self-test passed")


if __name__ == "__main__":
    self_test()
