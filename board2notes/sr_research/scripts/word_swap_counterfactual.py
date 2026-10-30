#!/usr/bin/env python3
"""Counterfactual word swap: WHICH words make PaddleOCR misread a far-away line?  (Lecture 8, slide 44)

The lecture moved the husky onto a lawn and watched the prediction flip. Here, for every far line we:
  1. read the far line as it is                                  -> raw CER
  2. replace EVERY word box with the same box from the clear photo -> "all clear" CER
     (also read the WHOLE clear line, to see how much error sits outside the word boxes: commas, underlines)
     (the best any perfect super-resolution of these words could do; raw - all clear = the gap)
  3. replace ONE word at a time with its clear version           -> how much fixing that word alone helps
  4. make everything clear EXCEPT one word                       -> how much that word alone still breaks the line

No model is trained and nothing needs gradients: PaddleOCR is used as a black box (forward passes only),
exactly like occlusion in the lecture. Needs `finetune_tpgsr_whiteboard.py prepare` to have been run (it
makes the word boxes, from the far photo only).

    python scripts/word_swap_counterfactual.py            # main test board (Airdrop B3) + 17 Oct test photos
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finetune_tpgsr_whiteboard as ft  # noqa: E402  (same word boxes, same CER, same PaddleOCR)

SETS = {"Airdrop B3 (unseen board)": lambda ln: ln["group"] == ft.TEST_GROUP,
        "17 Oct held-out photos": lambda ln: ln["source"] == "17oct" and ln["split"] == "test"}


def swapped(line, words, clear_ids, inp, tgt):
    """The far line with the word boxes in `clear_ids` replaced by the same boxes of the clear photo."""
    img = inp.copy()
    for wi in clear_ids:
        w = words[wi]
        img[w["y0"]:w["y1"], w["x0"]:w["x1"]] = tgt[w["y0"]:w["y1"], w["x0"]:w["x1"]]
    return img


def main() -> None:
    import pandas as pd
    _, _, lines, words = ft.load_prepared()
    rows, word_rows, examples = [], [], []
    for set_name, keep in SETS.items():
        ls = [ln for ln in lines if keep(ln) and ln["cer_eligible"] == "yes" and ln["label"] and ln["words"]]
        print(f"{set_name}: {len(ls)} lines")
        for k, ln in enumerate(ls, 1):
            inp, tgt = ft.read_rgb(ft.REPO / ln["input_path"]), ft.read_rgb(ft.REPO / ln["target_path"])
            ids = ln["words"]
            images = [inp, swapped(ln, words, ids, inp, tgt), tgt]                                 # far, words clear, whole clear line
            images += [swapped(ln, words, [wi], inp, tgt) for wi in ids]                          # fix one word
            images += [swapped(ln, words, [x for x in ids if x != wi], inp, tgt) for wi in ids]   # all but one
            reads = ft.paddle_read(images)
            c = [ft.line_cer(ln["label"], t) for t, _ in reads]
            raw, clear, full = c[0], c[1], c[2]
            fix_one, all_but = c[3:3 + len(ids)], c[3 + len(ids):]
            rows.append(dict(set=set_name, pair_id=ln["pair_id"], label=ln["label"], words=len(ids),
                             raw_cer=raw, all_clear_cer=clear, full_clear_cer=full, gap=raw - clear,
                             best_single_gain=max(raw - f for f in fix_one), raw_read=reads[0][0], clear_read=reads[1][0]))
            for j, wi in enumerate(ids):
                w = words[wi]
                word_rows.append(dict(set=set_name, pair_id=ln["pair_id"], label=ln["label"], word_index=j,
                                      native_h=w.get("native_h", np.nan), box_w=w["x1"] - w["x0"],
                                      gain_if_fixed_alone=raw - fix_one[j], damage_if_left_blurry=all_but[j] - clear,
                                      read_fixed_alone=reads[3 + j][0]))
            examples.append((set_name, ln, inp, images, reads, c))
            if k % 10 == 0:
                print(f"  {k}/{len(ls)}")
    df, wdf = pd.DataFrame(rows), pd.DataFrame(word_rows)
    df.to_csv(ft.OUT / "word_swap_lines.csv", index=False)
    wdf.to_csv(ft.OUT / "word_swap_words.csv", index=False)

    out = []
    for set_name, part in df.groupby("set", sort=False):
        wp = wdf[wdf.set == set_name]
        gap = part.gap.clip(lower=0).sum()
        out += [f"\n=== {set_name}: {len(part)} lines, {len(wp)} words ===",
                f"raw CER {part.raw_cer.mean():.3f} | every word from the clear photo {part.all_clear_cer.mean():.3f} "
                f"| gap (the most a perfect super-resolution of these words could win) {part.gap.mean():+.3f}",
                f"whole clear line (words AND the bits between them: commas, underlines, bullets) {part.full_clear_cer.mean():.3f} "
                f"-> {part.all_clear_cer.mean() - part.full_clear_cer.mean():+.3f} of the error lives outside the word boxes",
                f"lines that ANY word swap can improve: {(part.gap > 1e-9).sum()} of {len(part)}"]
        if gap > 0:
            share = part.best_single_gain.clip(lower=0).sum() / gap
            one_word = ((part.gap > 1e-9) & (part.best_single_gain >= 0.8 * part.gap)).sum()
            out += [f"fixing only the single most important word recovers {share:.0%} of the gap",
                    f"lines where ONE word holds >= 80% of the gap: {one_word} of {(part.gap > 1e-9).sum()}"]
            small = wp.native_h < 16
            if small.any() and (~small).any():
                out.append(f"mean gain from fixing one word: box < 16 px in the far photo {wp[small].gain_if_fixed_alone.mean():+.3f} "
                           f"({small.sum()} words) vs >= 16 px {wp[~small].gain_if_fixed_alone.mean():+.3f} ({(~small).sum()} words)")
            corr = np.corrcoef(wp.box_w, wp.gain_if_fixed_alone)[0, 1] if len(wp) > 2 and wp.gain_if_fixed_alone.std() > 0 else np.nan
            out.append(f"correlation of gain with word width (longer words = more letters): {corr:+.2f}")
        worst = wp.sort_values("damage_if_left_blurry", ascending=False).head(5)
        out.append("words that break an otherwise clear line the most (damage = CER added when only this word stays far):")
        for r in worst.itertuples():
            out.append(f"  {r.label!r} word {r.word_index + 1}: damage {r.damage_if_left_blurry:+.3f} (box {r.native_h:.0f} px tall in the far photo)")
    out += ["\nHow to read it:",
            "- 'Every word clear' much worse than 'whole clear line': the word boxes miss punctuation/strokes; fix boxes first.",
            "- Gap ~0: these lines are not limited by blur: super-resolution of the words cannot help them.",
            "- Most of the gap in ONE word per line: a targeted fix (better crop, retake prompt, that word's reading) is enough.",
            "- Gap spread over all words: every word is degraded; only a better capture (closer / zoom) or a reader trained on such",
            "  images can close it."]
    text = "\n".join(out)
    print(text)
    (ft.OUT / "word_swap_summary.txt").write_text(text)
    figure(examples, words)


def figure(examples, words, n=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    main = [e for e in examples if e[0].startswith("Airdrop")] or examples
    main = sorted(main, key=lambda e: -(e[5][0] - e[5][1]))[:n]          # the lines with the biggest gap
    if not main:
        return
    fig, axes = plt.subplots(len(main) * 3, 1, figsize=(12, 2.0 * 3 * len(main)))
    axes = np.atleast_1d(axes)
    for k, (_, ln, inp, images, reads, c) in enumerate(main):
        ids = ln["words"]
        best = int(np.argmax([c[0] - x for x in c[3:3 + len(ids)]]))
        panels = [(images[0], f"far line | truth {ln['label']!r} | PaddleOCR {reads[0][0]!r} | CER {c[0]:.2f}"),
                  (images[3 + best], f"only word {best + 1} from the clear photo | {reads[3 + best][0]!r} | CER {c[3 + best]:.2f}"),
                  (images[1], f"every word from the clear photo | {reads[1][0]!r} | CER {c[1]:.2f}")]
        for j, (im, title) in enumerate(panels):
            a = axes[3 * k + j]
            a.imshow(im)
            for i, wi in enumerate(ids):
                w = words[wi]
                a.add_patch(plt.Rectangle((w["x0"], w["y0"]), w["x1"] - w["x0"], w["y1"] - w["y0"], fill=False,
                                          ec="lime" if (j == 1 and i == best) or j == 2 else "red", lw=0.8))
            a.set_title(title, fontsize=8, loc="left"); a.axis("off")
    plt.tight_layout()
    fig.savefig(ft.OUT / "word_swap_examples.png", dpi=120)
    plt.close(fig)
    print(f"saved {(ft.OUT / 'word_swap_examples.png').relative_to(ft.REPO)} (red box = far word, green = swapped in from the clear photo)")


if __name__ == "__main__":
    main()
