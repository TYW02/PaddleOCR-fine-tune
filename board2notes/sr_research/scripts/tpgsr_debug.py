#!/usr/bin/env python3
"""Model debug for TPGSR on our whiteboard words: WHY did it not help PaddleOCR?   (Lecture 8: debug step)

Forward passes only, no training. Uses the model kept by finetune_tpgsr_whiteboard.py (epoch 0 = the TextZoom
model) and its word pairs. Four parts, each answers one question:

  A  prior    Is the text prior the bottleneck?  Same words, same network, 5 different priors:
              CRNN on the far word (normal) | CRNN on the clear word | the TRUE label | a WRONG label | no letters.
              If the true label makes the words readable, the prior is to blame; if nothing changes, the
              network ignores its prior (and cannot sharpen on its own).
  B  reach    How far does each output pixel "see"?  Blank a 4 px strip of the input and measure how far
              away the output changes (occlusion, Lecture 8). In theory the GRUs see the whole word; this
              measures the EFFECTIVE reach, with the prior fixed (network only) and recomputed (as in use).
  C  inside   What happens layer by layer on one word (feature maps, the prior, the final change), and, over
              all B3 words, where the change goes: on strokes, next to strokes, or on the board.
  D  colour   Where the pink tint / visible box edges come from, and whether they cost readability.

    python scripts/tpgsr_debug.py            # all parts -> outputs/tpgsr_debug/
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finetune_tpgsr_whiteboard as ft  # noqa: E402  (same data, CER, PaddleOCR, model loading)

OUT = ft.REPO / "outputs" / "tpgsr_debug"
MAX_H = ft.Settings().sr_max_h          # restore every word below 32 px: we debug the network, not the gate
PRIORS = ["CRNN on far word (normal)", "CRNN on clear word", "true label", "wrong label", "no letters (all blank)"]


# ----------------------------------------------------------------------------------------------- setup
def setup():
    """Model, data and the labelled small words of B3 (main) and the synthetic validation lines."""
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    tz, gen, rec = ft.models(device := ft.tz_device())
    ck_path = ft.BEST_CHECKPOINT if ft.BEST_CHECKPOINT.exists() else ft.BASE_CHECKPOINT
    ck = torch.load(ck_path, map_location=device, weights_only=True)
    gen.load_state_dict(ck["generator"])
    gen.eval()
    print(f"model: {ck_path.relative_to(ft.REPO)} (epoch {ck.get('epoch')}) on {device}")
    lr_all, hr_all, lines, words = ft.load_prepared()
    sets = {"B3": lambda ln: ln["group"] == ft.TEST_GROUP,
            "synthetic val": lambda ln: ln["source"] == "airdrop_synth" and ln["split"] == "val"}
    data = {}
    for name, keep in sets.items():
        ls = [ln for ln in lines if keep(ln) and ln["label"] and ln["words"] and ln["cer_eligible"] == "yes"
              and ft.word_labels(ln) is not None]
        ws = [dict(word=wi, label=lab, native_h=words[wi]["native_h"], line=ln["pair_id"])
              for ln in ls for wi, lab in zip(ln["words"], ft.word_labels(ln))]
        data[name] = SimpleNamespace(lines=ls, words=ws)
        print(f"{name}: {len(ls)} lines with one label token per word box, {len(ws)} words "
              f"({sum(w['native_h'] < MAX_H for w in ws)} under {MAX_H:g} px)")
    T = tz.text_prior(rec, ft.to_tensor(lr_all[:1]).to(device)).shape[-1]
    return SimpleNamespace(tz=tz, gen=gen, rec=rec, device=device, lr=lr_all, hr=hr_all, lines=lines,
                           words=words, data=data, T=int(T))


# ----------------------------------------------------------------------------------------------- model helpers
def label_prior(ctx, labels):
    """One-hot 'perfect' priors: each letter at its share of the T time steps, CTC blank (index 0) elsewhere.
    An empty label (e.g. '&', not in the CRNN alphabet) gives the all-blank prior."""
    p = np.zeros((len(labels), len(ft.CRNN_CHARS), 1, ctx.T), np.float32)
    p[:, 0] = 1.0
    for i, lab in enumerate(labels):
        lab = [c for c in lab if c in ft.CRNN_CHARS[1:]][: ctx.T]
        for k, c in enumerate(lab):
            t = min(ctx.T - 1, int((k + 0.5) * ctx.T / len(lab)))
            p[i, :, 0, t] = 0.0
            p[i, ft.CRNN_CHARS.index(c), 0, t] = 1.0
    return p


def prior_of(ctx, kind, lr_u8, hr_u8, labels, wrong):
    """The prior as a numpy array (N, 37, 1, T) for one of the PRIORS."""
    import torch
    if kind == PRIORS[0]:
        x = ft.to_tensor(lr_u8).to(ctx.device)
    elif kind == PRIORS[1]:
        x = ft.to_tensor(hr_u8).to(ctx.device)
    else:
        return label_prior(ctx, {PRIORS[2]: labels, PRIORS[3]: wrong, PRIORS[4]: [""] * len(labels)}[kind])
    with torch.no_grad():
        return ctx.tz.text_prior(ctx.rec, x).cpu().numpy()


def run(ctx, lr_u8, prior, bs=64):
    """Generator on LR words (uint8) with a given prior (numpy) -> restored words, float RGB in [0, 1]."""
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(lr_u8), bs):
            lr = ft.to_tensor(lr_u8[i:i + bs]).to(ctx.device)
            pr = torch.from_numpy(np.ascontiguousarray(prior[i:i + bs])).to(ctx.device)
            sr = ((ctx.gen(lr, pr) + 1.0) / 2.0).clamp(0, 1)
            out.append(sr.permute(0, 2, 3, 1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, ft.HR_SIZE[1], ft.HR_SIZE[0], 3), np.float32)


def _plt():
    """pyplot; the file-only 'Agg' backend when run as a script (inside Jupyter the notebook's backend is kept)."""
    import matplotlib
    if "ipykernel" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def u8(x):
    return (np.clip(x, 0, 1) * 255).round().astype(np.uint8)


def trace(ctx, lr_u8, prior):
    """TSRN_TL.forward written out step by step (same order as TPGSR's model/tsrn.py), keeping every block.
    Returns numpy arrays: block1..block7 (N, 64, 16, 64), prior_map (N, 32, 16, 64), pre_tanh (N, 3, 32, 128),
    output in [0, 1] (N, 32, 128, 3), and `same_as_model`: the largest difference to calling the model directly."""
    import torch
    import torch.nn.functional as F
    g = ctx.gen
    with torch.no_grad():
        x = ft.to_tensor(lr_u8).to(ctx.device)
        p = torch.from_numpy(np.ascontiguousarray(prior)).to(ctx.device)
        b = {"1": g.block1(x)}
        emb = F.interpolate(g.infoGen(p), (x.shape[2], x.shape[3]), mode="bilinear", align_corners=True)
        for i in range(g.srb_nums + 1):
            k = i + 2
            blk = getattr(g, f"block{k}")
            b[str(k)] = blk(b[str(k - 1)], emb) if k in (2, 3, 4, 5, 6) else blk(b[str(k - 1)])
        pre = getattr(g, f"block{g.srb_nums + 3}")(b["1"] + b[str(g.srb_nums + 2)])
        out = ((torch.tanh(pre) + 1.0) / 2.0).clamp(0, 1)
        direct = ((g(x, p) + 1.0) / 2.0).clamp(0, 1)
        same = float((out - direct).abs().max().cpu().numpy())
    res = {f"block{k}": b[k].cpu().numpy() for k in b}
    res.update(prior_map=emb.cpu().numpy(), pre_tanh=pre.cpu().numpy(),
               output=out.permute(0, 2, 3, 1).cpu().numpy(), same_as_model=same)
    return res


def ink_mask(hr_u8):
    """Ink of the CLEAR word (N, 32, 128): clearly darker than its own board (adaptive, as in word_boxes)."""
    g = hr_u8.astype(np.float32).mean(axis=3) / 255.0
    board = np.percentile(g.reshape(len(g), -1), 80, axis=1)[:, None, None]
    dark = np.percentile(g.reshape(len(g), -1), 2, axis=1)[:, None, None]
    return g < board - 0.45 * (board - dark)


def bands(hr_u8):
    """stroke / near stroke (within 2 px) / board, from the clear word."""
    ink = ink_mask(hr_u8)
    near = np.stack([cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0 for m in ink]) & ~ink
    return {"stroke": ink, "near stroke (2 px)": near, "board": ~(ink | near)}


def bicubic(lr_u8):
    return ft.bicubic_words(lr_u8).astype(np.float32) / 255.0


def paste_lines(ctx, ls, word_ids, word_imgs_u8):
    """Line CER when the given words are replaced (every word < MAX_H px; others stay raw)."""
    full = np.zeros((len(ctx.lr), ft.HR_SIZE[1], ft.HR_SIZE[0], 3), np.uint8)
    full[word_ids] = word_imgs_u8
    return ft.lines_cer(ls, full, ctx.words, max_h=MAX_H)


def far_crop(ctx, wi):
    w = ctx.words[wi]
    ln = ctx.lines[w["line"]]
    img = ft.read_rgb(ft.REPO / ln["input_path"])
    return img[w["y0"]:w["y1"], w["x0"]:w["x1"]]


def wrong_labels(labels):
    """For each word, the label of another word with a different text (the counterfactual prior)."""
    n, out = len(labels), []
    for i, lab in enumerate(labels):
        j = (i + 1) % n
        while n > 1 and labels[j] == lab and j != i:
            j = (j + 1) % n
        out.append(labels[j])
    return out


# ----------------------------------------------------------------------------------------------- part A
def part_a_prior(ctx):
    """5 priors x the same words: PaddleOCR word CER, line CER, pixel error, and how much the prior moves the image."""
    import pandas as pd
    rows, lines_rows, examples = [], [], {}
    for set_name, d in ctx.data.items():
        ws = [w for w in d.words if w["native_h"] < MAX_H]
        if not ws:
            continue
        ids = np.array([w["word"] for w in ws])
        labels = [w["label"] for w in ws]
        wrong = wrong_labels(labels)
        lr, hr = ctx.lr[ids], ctx.hr[ids]
        outs = {k: run(ctx, lr, prior_of(ctx, k, lr, hr, labels, wrong)) for k in PRIORS}
        refs = {"far word as photographed": [far_crop(ctx, wi) for wi in ids],
                "16px path, no network": list(u8(bicubic(lr))),
                "clear photo": list(hr)}
        imgs = {**refs, **{k: list(u8(v)) for k, v in outs.items()}}
        reads = {k: ft.paddle_read(v) for k, v in imgs.items()}
        hr_f = hr.astype(np.float32) / 255.0
        for k, v in imgs.items():
            for j, w in enumerate(ws):
                o = outs.get(k)
                rows.append(dict(set=set_name, version=k, word=w["label"], native_h=w["native_h"],
                                 read=reads[k][j][0], confidence=reads[k][j][1],
                                 word_cer=ft.line_cer(w["label"], reads[k][j][0]) if w["label"] else np.nan,
                                 pixel_error=float(np.abs(o[j] - hr_f[j]).mean() * 255) if o is not None else np.nan,
                                 moved_vs_normal=float(np.abs(o[j] - outs[PRIORS[0]][j]).mean() * 255) if o is not None else np.nan,
                                 prior_given=(wrong[j] if k == PRIORS[3] else w["label"] if k == PRIORS[2] else "")))
        # whole lines, every small word replaced
        raw_c = ft.lines_cer(d.lines, None, ctx.words)[0]
        lines_rows.append(dict(set=set_name, version="far word as photographed", cer=float(np.mean(raw_c))))
        for k in ["16px path, no network", *PRIORS, "clear photo"]:
            src = u8(bicubic(lr)) if k == "16px path, no network" else hr if k == "clear photo" else u8(outs[k])
            c = paste_lines(ctx, d.lines, ids, src)[0]
            lines_rows.append(dict(set=set_name, version=k, cer=float(np.mean(c))))
        examples[set_name] = (ws, imgs, reads, wrong)
    df, ldf = pd.DataFrame(rows), pd.DataFrame(lines_rows)
    df.to_csv(OUT / "a_prior_words.csv", index=False)
    ldf.to_csv(OUT / "a_prior_lines.csv", index=False)
    table = (df.groupby(["set", "version"], sort=False)
               .agg(word_cer=("word_cer", "mean"), exact=("word_cer", lambda x: (x == 0).mean()),
                    pixel_error=("pixel_error", "mean"), moved_vs_normal=("moved_vs_normal", "mean")).reset_index())
    table = table.merge(ldf.rename(columns={"cer": "line_cer"}), on=["set", "version"], how="left")
    figure_a(examples)
    return table, df


def figure_a(examples, n=6):
    plt = _plt()
    if "B3" not in examples:
        return
    ws, imgs, reads, wrong = examples["B3"]
    order = sorted(range(len(ws)), key=lambda j: ws[j]["native_h"])[:n]          # the smallest words
    cols = ["far word as photographed", "16px path, no network", *PRIORS, "clear photo"]
    fig, axes = plt.subplots(len(order), len(cols), figsize=(2.3 * len(cols), 1.05 * len(order) + 0.6))
    axes = np.atleast_2d(axes)
    for r, j in enumerate(order):
        for c, k in enumerate(cols):
            a = axes[r, c]
            a.imshow(cv2.resize(imgs[k][j], ft.HR_SIZE, interpolation=cv2.INTER_CUBIC)); a.set_xticks([]); a.set_yticks([])
            extra = f"\nprior: {wrong[j]!r}" if k == PRIORS[3] else ""
            a.set_xlabel(f"{reads[k][j][0]!r}{extra}", fontsize=7)
            if r == 0:
                a.set_title(k.replace(" (", "\n("), fontsize=7)
        axes[r, 0].set_ylabel(f"{ws[j]['label']!r}\n{ws[j]['native_h']:.0f} px", fontsize=7)
    fig.suptitle("Part A: the same B3 words restored with 5 different text priors (PaddleOCR reading under each)", fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT / "a_prior_examples.png", dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- part B
def part_b_reach(ctx, strip=4, max_words=40):
    """Occlusion: blank a strip of the LR input, measure the output change at every distance from it."""
    import pandas as pd
    ws = [w for w in ctx.data["B3"].words if w["native_h"] < MAX_H][:max_words]
    ids = np.array([w["word"] for w in ws])
    lr, hr = ctx.lr[ids], ctx.hr[ids]
    labels = [w["label"] for w in ws]
    base_prior = prior_of(ctx, PRIORS[0], lr, hr, labels, labels)
    base = run(ctx, lr, base_prior)
    base_bic = bicubic(lr)
    W = lr.shape[2]
    positions = list(range(0, W - strip + 1, strip))
    prof = {m: np.zeros(W) for m in ("network only (prior fixed)", "as used (prior recomputed)", "bicubic, no network")}
    cnt = np.zeros(W)
    heat = {m: np.zeros((len(positions), ft.HR_SIZE[0])) for m in prof}
    out_col_lr = (np.arange(ft.HR_SIZE[0]) + 0.5) / 2.0                 # output column -> LR column
    for pi, x0 in enumerate(positions):
        occ = lr.copy()
        fill = np.median(lr.reshape(len(lr), -1, 3), axis=1).astype(np.uint8)
        occ[:, :, x0:x0 + strip] = fill[:, None, None, :]
        outs = {"network only (prior fixed)": run(ctx, occ, base_prior),
                "as used (prior recomputed)": run(ctx, occ, prior_of(ctx, PRIORS[0], occ, hr, labels, labels)),
                "bicubic, no network": bicubic(occ)}
        d = (np.abs(np.floor(out_col_lr) - x0 - (strip - 1) / 2.0)).astype(int)     # distance in LR px from the strip
        for m, o in outs.items():
            ref = base_bic if m.startswith("bicubic") else base
            change = np.abs(o - ref).mean(axis=(0, 1, 3)) * 255                      # per output column, grey levels
            heat[m][pi] = change
            np.add.at(prof[m], d, change)
        np.add.at(cnt, d, 1)
    rows, summary = [], {}
    for m in prof:
        p = prof[m][cnt > 0] / cnt[cnt > 0]
        dist = np.nonzero(cnt > 0)[0]
        # influence "mass" at each distance: both sides of the strip count (one column at distance 0)
        mass = p * np.where(dist == 0, 1, 2)
        share = np.cumsum(mass) / max(mass.sum(), 1e-12)
        r90 = int(dist[np.searchsorted(share, 0.9)]) if len(dist) else 0
        beyond = float(mass[dist > 8].sum() / max(mass.sum(), 1e-12))
        summary[m] = dict(radius_90=r90, beyond_one_letter=beyond, change_at_strip=float(p[0]) if len(p) else 0.0)
        rows += [dict(mode=m, distance=int(x), change=float(v), radius_90=r90, beyond_one_letter=beyond)
                 for x, v in zip(dist, p)]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "b_reach.csv", index=False)
    figure_b(df, heat, positions, ws, lr, strip)
    return summary, df


def figure_b(df, heat, positions, ws, lr, strip):
    plt = _plt()
    fig = plt.figure(figsize=(13, 7.5))
    ax = fig.add_subplot(2, 1, 1)
    for m, part in df.groupby("mode", sort=False):
        ax.plot(part.distance, part.change, marker="o", ms=3,
                label=f"{m}: 90% of the influence within {part.radius_90.iloc[0]} px, "
                      f"{part.beyond_one_letter.iloc[0]:.0%} beyond 1 letter")
    ax.axvline(8, ls=":", c="grey")
    ax.text(8.3, 0.5, "~1 letter (8 LR px)", fontsize=8, color="grey", transform=ax.get_xaxis_transform())
    ax.set_xlabel("distance from the blanked strip (LR pixels; the word is 64 px wide, 16 px tall)")
    ax.set_ylabel("mean output change (grey levels)")
    ax.set_title(f"Part B: how far does blanking {strip} px of the input change the output? ({len(ws)} B3 words)", fontsize=10)
    ax.set_yscale("symlog", linthresh=0.1); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    for i, m in enumerate(["network only (prior fixed)", "bicubic, no network"]):
        a = fig.add_subplot(2, 2, 3 + i)
        a.imshow(heat[m], aspect="auto", cmap="magma", extent=[0, lr.shape[2], positions[-1] + strip, 0])
        a.set_xlabel("output position (in LR pixels)"); a.set_ylabel("blanked strip starts at")
        a.set_title(f"{m}: change map (bright = changed)", fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT / "b_reach.png", dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- part C
def part_c_inside(ctx, word=None):
    """One word layer by layer, plus (all B3 words) how much the prior changes each block and where the change goes."""
    import pandas as pd
    ws = [w for w in ctx.data["B3"].words if w["native_h"] < MAX_H]
    pick = next((w for w in ws if word and word in w["label"]), None) or \
        min((w for w in ws if len(w["label"]) >= 5), key=lambda w: w["native_h"], default=ws[0])
    ids = np.array([w["word"] for w in ws])
    lr, hr = ctx.lr[ids], ctx.hr[ids]
    labels = [w["label"] for w in ws]
    t_norm = trace(ctx, lr, prior_of(ctx, PRIORS[0], lr, hr, labels, labels))
    t_true = trace(ctx, lr, prior_of(ctx, PRIORS[2], lr, hr, labels, labels))
    print(f"sanity check: step-by-step trace vs calling the model: largest difference {t_norm['same_as_model']:.2e} "
          f"({'OK' if t_norm['same_as_model'] < 1e-4 else 'MISMATCH: the trace does not match the model'})")
    # how much does the TRUE label change each block (relative to the block's own size)?
    blocks = [k for k in t_norm if k.startswith("block")]
    effect = [dict(block=k, prior_effect=float(np.linalg.norm(t_true[k] - t_norm[k]) / (np.linalg.norm(t_norm[k]) + 1e-8)))
              for k in blocks]
    # where does the network change the picture? (vs the 16px path), and does it get closer to the clear word?
    bic, out, hrf = bicubic(lr), t_norm["output"], hr.astype(np.float32) / 255.0
    rows = []
    for name, m in bands(hr).items():
        mm = np.broadcast_to(m[..., None], out.shape)
        sgn = (out - bic).mean(axis=3)[m]
        rows.append(dict(band=name, share_of_pixels=float(m.mean()),
                         mean_change=float(sgn.mean() * 255),                      # + = lighter, - = darker
                         error_16px_path=float(np.abs(bic - hrf)[mm].mean() * 255),
                         error_tpgsr=float(np.abs(out - hrf)[mm].mean() * 255)))
    j = int(np.nonzero(ids == pick["word"])[0][0])
    figure_c(ctx, pick, j, t_norm, t_true, lr, hr, bic, labels)
    return pd.DataFrame(effect), pd.DataFrame(rows), pick


def figure_c(ctx, pick, j, t_norm, t_true, lr, hr, bic, labels):
    plt = _plt()
    blocks = [k for k in t_norm if k.startswith("block")]
    read = ft.crnn_read(ctx.tz, ctx.rec, lr[j:j + 1], ctx.device)[0]
    panels = [("input (far word, 16 x 64)", lr[j], None),
              (f"prior map (mean of 32 channels): CRNN reads {read!r}", t_norm["prior_map"][j].mean(0), "viridis"),
              ("prior map with the TRUE label", t_true["prior_map"][j].mean(0), "viridis")]
    panels += [(f"{k}: mean |activation|", np.abs(t_norm[k][j]).mean(0), "magma") for k in blocks]
    change = (t_norm["output"][j] - bic[j]).mean(2)
    lim = max(1e-3, np.abs(change).max())
    panels += [("output (TPGSR)", u8(t_norm["output"][j]), None),
               ("change vs 16px path: blue = darker, red = lighter", change, "bwr"),
               ("clear photo (target)", hr[j], None)]
    fig, axes = plt.subplots(len(panels), 1, figsize=(7, 1.25 * len(panels)))
    for a, (title, im, cmap) in zip(axes, panels):
        kw = dict(vmin=-lim, vmax=lim) if cmap == "bwr" else {}
        a.imshow(im, cmap=cmap, aspect="auto", **kw)
        a.set_title(title, fontsize=8, loc="left"); a.axis("off")
        if cmap == "bwr":
            a.contour(ink_mask(hr[j:j + 1])[0], levels=[0.5], colors="k", linewidths=0.4)
    fig.suptitle(f"Part C: {pick['label']!r} ({pick['native_h']:.0f} px tall in the far photo), layer by layer", fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT / "c_layers.png", dpi=130)
    plt.close(fig)
    # the prior itself: 37 characters x T time steps
    p_norm = prior_of(ctx, PRIORS[0], lr[j:j + 1], hr[j:j + 1], [labels[j]], [labels[j]])[0, :, 0]
    p_true = label_prior(ctx, [labels[j]])[0, :, 0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for a, (title, p) in zip(axes, [(f"CRNN prior from the far word (reads {read!r})", p_norm),
                                    (f"true-label prior ({labels[j]!r})", p_true)]):
        a.imshow(p, aspect="auto", cmap="Greys", vmin=0, vmax=1)
        a.set_yticks(range(len(ft.CRNN_CHARS))); a.set_yticklabels(["blank"] + list(ft.CRNN_CHARS[1:]), fontsize=6)
        a.set_xlabel("time step (left -> right across the word)"); a.set_title(title, fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT / "c_prior_matrix.png", dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- part D
def part_d_colour(ctx, n_textzoom=200):
    """Colour shift of the restored word vs the 16px path, on board pixels; and the CER once the shift is removed."""
    import pandas as pd
    rows = []
    sources = {}
    for set_name, d in ctx.data.items():
        ws = [w for w in d.words if w["native_h"] < MAX_H]
        if ws:
            ids = np.array([w["word"] for w in ws])
            sources[set_name] = (ctx.lr[ids], ctx.hr[ids], ids, d.lines)
    try:                                          # TextZoom pairs (what the model was trained on): the reference
        tzd = ctx.tz.TextZoomPairs(ft.REPO / "data" / "textzoom" / "train2")
        pick = np.linspace(0, len(tzd) - 1, min(n_textzoom, len(tzd))).astype(int)
        pairs = [tzd[int(i)] for i in pick]
        lr_t = np.stack([u8(p[0].permute(1, 2, 0).cpu().numpy()) for p in pairs])
        hr_t = np.stack([u8(p[1].permute(1, 2, 0).cpu().numpy()) for p in pairs])
        sources["TextZoom (training data)"] = (lr_t, hr_t, None, None)
    except Exception as e:                        # noqa: BLE001  (TextZoom missing: skip the reference)
        print(f"TextZoom reference skipped: {e}")
    fixed = {}
    for name, (lr, hr, ids, ls) in sources.items():
        out = run(ctx, lr, prior_of(ctx, PRIORS[0], lr, hr, [""] * len(lr), [""] * len(lr)))
        bic = bicubic(lr)
        board = bands(hr)["board"]
        for c, ch in enumerate("RGB"):
            rows.append(dict(set=name, channel=ch,
                             board_shift=float((out[..., c] - bic[..., c])[board].mean() * 255),
                             board_level_16px=float(bic[..., c][board].mean() * 255)))
        if ids is not None:                       # remove each word's own board shift, per channel
            shift = np.stack([[(out[i, ..., c] - bic[i, ..., c])[board[i]].mean() if board[i].any() else 0.0
                               for c in range(3)] for i in range(len(out))])
            fixed[name] = (ids, ls, u8(out), u8(out - shift[:, None, None, :]))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "d_colour.csv", index=False)
    cer_rows = []
    for name, (ids, ls, plain, matched) in fixed.items():
        for version, imgs in [("TPGSR as is", plain), ("TPGSR, board colour matched", matched)]:
            cer_rows.append(dict(set=name, version=version, line_cer=float(np.mean(paste_lines(ctx, ls, ids, imgs)[0]))))
    return df, pd.DataFrame(cer_rows)


# ----------------------------------------------------------------------------------------------- main
def summary_text(a, reach, effect, where, pick, colour, cer):
    out = ["=== A. Is the text prior the bottleneck? (words < 32 px; CER lower = better) ===", a.round(3).to_string(index=False)]
    out += ["\n=== B. Effective reach (LR px; a letter is ~8 px, the word 64 px; bicubic = what 'purely local' looks like) ===",
            *[f"  {m}: 90% of the influence within {r['radius_90']} px | {r['beyond_one_letter']:.0%} of it beyond one letter "
              f"(8 px) | change at the strip {r['change_at_strip']:.2f} grey levels" for m, r in reach.items()]]
    out += [f"\n=== C. Inside the network (B3 words; example word {pick['label']!r}) ===",
            "how much the TRUE label changes each block (relative size):", effect.round(3).to_string(index=False),
            "where the network changes the picture vs the 16px path (grey levels; + lighter, - darker):",
            where.round(2).to_string(index=False)]
    out += ["\n=== D. Colour shift on board pixels (grey levels, TPGSR - 16px path) ===", colour.round(2).to_string(index=False),
            "line CER with and without the shift:", cer.round(3).to_string(index=False)]
    text = "\n".join(out)
    (OUT / "summary.txt").write_text(text)
    return text


def main():
    import pandas as pd
    pd.set_option("display.width", 200)
    ctx = setup()
    a, _ = part_a_prior(ctx)
    reach, _ = part_b_reach(ctx)
    effect, where, pick = part_c_inside(ctx)
    colour, cer = part_d_colour(ctx)
    print(summary_text(a, reach, effect, where, pick, colour, cer))
    print(f"saved figures and tables in {OUT.relative_to(ft.REPO)}")


if __name__ == "__main__":
    main()
