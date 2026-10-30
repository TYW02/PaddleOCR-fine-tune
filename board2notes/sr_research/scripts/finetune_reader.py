#!/usr/bin/env python3
"""Fine-tune the pipeline's own reader (PaddleOCR PP-OCRv6_medium_rec) on our whiteboard handwriting.

Why: on the unseen CNN board (B3), PaddleOCR misreads even the CLEAR photo (line CER 0.068: "Architectre",
"Akxnet", "CUN"), and on the 17 Oct board almost all of its error remains in clear photos. That is the
reader not knowing marker handwriting, which no image restoration can fix (docs/SUPER_RESOLUTION_RUNS.md,
sections 10-12). Fine-tuning the reader on our labelled line crops attacks it directly, and any gain goes
straight into the app (same model, same API).

Recipe: PaddleOCR's own (docs/version3.x/module_usage/text_recognition.en.md, "Secondary Development"):
tools/train.py from the official pretrained weights -> tools/export_model.py -> TextRecognition(model_dir=...).

Data (the same split as scripts/train_line_recognizer.py, so results stay comparable):
  train  our crops of Airdrop boards B1, B4, B5 and of the 17 Oct photos outside the validation / test photos
         (repeated 4x per epoch) + IAM handwritten lines; only lines of at most 25 characters, because in training
         the model squeezes every line into 40 time steps (see Settings). Only 'accepted' / 'duplicate' crops:
         'review' crops can have labels the crop does not show.
  val    Airdrop B2 + the 17 Oct validation photos: chooses the epoch (norm_edit_dis = 1 - normalised edit
         distance, the closest of PaddleOCR's metrics to our CER).
  test   Airdrop B3 (the CNN board, never trained on: 7 sentences, 26 photos) = the decision;
         the 17 Oct test photos (only a few crops: reported); IAM test (did it forget general handwriting?).

Pass rule, fixed before training: on B3 the fine-tuned reader must have a lower mean CER than the pretrained
one AND more crops read better than worse; the IAM test CER may not rise by more than 0.02.

    python scripts/finetune_reader.py prepare|config|baseline|train|export|evaluate|select|all [--smoke]
    comparison (scripts/runpod/run_cloud.sh): train/export/evaluate --val-only --run A|B|C, then select --runs A,B,C
    (baseline = diagnosis: the untrained pretrained weights on the validation lines, via tools/eval.py)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reading_fusion import edit_distance, normalise  # noqa: E402  (the CER's text normalisation)

REPO = Path(__file__).resolve().parents[1]
ASSIGNMENT = REPO.parent                                     # data_dir of every label file
PADDLEOCR = ASSIGNMENT / "models" / "PaddleOCR-text-sr"      # PaddleOCR source checkout (tools/train.py)
BASE_CONFIG = PADDLEOCR / "configs" / "rec" / "PP-OCRv6" / "PP-OCRv6_medium_rec.yml"
DICT = PADDLEOCR / "ppocr" / "utils" / "dict" / "ppocrv6_dict.txt"
PRETRAINED_URL = ("https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/"
                  "PP-OCRv6_medium_rec_pretrained.pdparams")
MODEL_NAME = "PP-OCRv6_medium_rec"
MODELS = REPO / "models" / "reader"
PRETRAINED = MODELS / "PP-OCRv6_medium_rec_pretrained.pdparams"
OUT = REPO / "outputs" / "reader_finetune"
EXPORT = MODELS / "PP-OCRv6_medium_rec_whiteboard_infer"
AIRDROP = ASSIGNMENT / "airdrop_rec_paddle"
OCT17 = ASSIGNMENT / "17oct_rec_paddle"
IAM = ASSIGNMENT / "iam_line_paddle"
VAL_BOARDS, TEST_BOARDS = {"B2"}, {"B3"}
OCT17_VAL_PHOTOS = {"IMG_4736", "IMG_4705", "IMG_4673"}     # as in pair_data / train_line_recognizer
OCT17_TEST_PHOTOS = {"IMG_4748", "IMG_4750", "IMG_4697"}
FILES = ["train_own", "train_iam", "val", "test_b3", "test_17oct", "test_iam"]


@dataclass
class Settings:
    epochs: int = 10
    learning_rate: float = 1e-4        # 5x below the from-scratch 5e-4: fine-tuning
    warmup_epochs: int = 1
    batch_size: int = 16               # CPU-friendly; on a GPU (Kaggle) 64 is fine
    # Training geometry = PaddleOCR's own recipe. In TRAINING mode PPLCNetV4 pools every line to exactly 40
    # time steps (rec_lcnetv4.py: adaptive_avg_pool2d(x, [1, 40])), whatever the width, so a CTC label must be
    # well under 40 characters. The first smoke test used 640 px / 100 characters (65% of IAM lines > 40
    # characters): impossible alignments taught the reader to stop emitting blanks ('e e ee'), 21 Oct.
    train_width: int = 320             # official training width (8 px per time step, as when reading)
    max_text_length: int = 25          # official; longer TRAINING lines are dropped (validation / test keep all)
    eval_width: int = 640              # validation is read like inference (width / 8 time steps): 0.98 untrained
    own_repeat: int = 4                # our crops appear 4x per epoch
    iam_ratio: float = 1.0             # IAM lines that fit (<= 25 characters: ~300) all every epoch
    num_workers: int = 0               # 0 is safest on macOS; 4 on Linux / Kaggle
    freeze_bn: bool = True             # batch-norm keeps its pretrained statistics (scripts/paddle_train_frozen_bn.py):
                                       # without it 10 steps of batch 8 wrecked the reader (smoke test, 21 Oct)
    freeze_backbone: bool = False      # train only the neck + heads (the backbone keeps its pretrained weights)
    run: str = ""                      # name of one setting in a comparison: its own folder outputs/reader_finetune/<run>/
    seed: int = 20261021
    smoke: bool = False


# ----------------------------------------------------------------------------------------------- text
def cer(reference: str, prediction: str) -> float:
    """The team's line CER: lower case, single spaces, empty reading -> ' ', capped at 1 (no jiwer needed)."""
    ref, hyp = normalise(reference), normalise(prediction) or " "
    return min(edit_distance(ref, hyp) / max(len(ref), 1), 1.0)


def dict_chars() -> set:
    return {line.rstrip("\n") for line in DICT.open(encoding="utf-8")} | {" "}


# ----------------------------------------------------------------------------------------------- prepare
def own_rows():
    """(role, relative path, label, photo, source) for every usable whiteboard crop."""
    out = []
    for r in csv.DictReader((AIRDROP / "manifest.csv").open(encoding="utf-8")):
        if r["status"] not in ("accepted", "duplicate") or not r["label"].strip():
            continue
        role = "test_b3" if r["board"] in TEST_BOARDS else "val" if r["board"] in VAL_BOARDS else "train_own"
        out.append((role, f"{AIRDROP.name}/{r['crop']}", r["label"].strip(), r["photo"], f"airdrop {r['board']}"))
    for r in csv.DictReader((OCT17 / "manifest.csv").open(encoding="utf-8")):
        if r["status"] != "accepted" or not r["label"].strip():
            continue
        role = "val" if r["photo"] in OCT17_VAL_PHOTOS else "test_17oct" if r["photo"] in OCT17_TEST_PHOTOS else "train_own"
        out.append((role, f"{OCT17.name}/{r['crop']}", r["label"].strip(), r["photo"], "17oct"))
    return out


def iam_rows():
    out = []
    for split, role in (("train", "train_iam"), ("test", "test_iam")):
        for line in (IAM / f"{split}.txt").open(encoding="utf-8"):
            path, _, label = line.rstrip("\n").partition("\t")
            if label.strip():
                out.append((role, f"{IAM.name}/{path}", label.strip(), path.split("/")[-1].split("-")[0], "iam"))
    return out


def base_dir(s: Settings) -> Path:
    return OUT.parent / "reader_finetune_smoke" if s.smoke else OUT


def data_dir(s: Settings) -> Path:            # shared by every run
    return base_dir(s) / "data"


def run_dir(s: Settings) -> Path:             # one folder per setting of a comparison
    return base_dir(s) / s.run if s.run else base_dir(s)


def export_dir(s: Settings) -> Path:
    return run_dir(s) / "inference" if (s.smoke or s.run) else EXPORT


def prepare(s: Settings) -> None:
    d = data_dir(s)
    d.mkdir(parents=True, exist_ok=True)
    chars = dict_chars()
    rows = own_rows() + iam_rows()
    dropped = Counter()
    keep = []
    for role, path, label, photo, source in rows:
        label = " ".join(label.replace("\t", " ").split())
        bad = {c for c in label if c not in chars}
        if bad:
            dropped["".join(sorted(bad))] += 1
            continue
        if role.startswith("train") and len(label) > s.max_text_length:
            dropped[f"{role}: longer than {s.max_text_length} characters (training only)"] += 1
            continue
        if not (ASSIGNMENT / path).exists():
            dropped["image missing"] += 1
            continue
        keep.append((role, path, label, photo, source))
    # leak checks: the test board and the val / test photos never reach training
    train = [r for r in keep if r[0].startswith("train")]
    assert not any(r[4] == "airdrop B3" or r[4] == "airdrop B2" for r in train), "B2/B3 leaked into training"
    assert not any(r[4] == "17oct" and r[3] in OCT17_VAL_PHOTOS | OCT17_TEST_PHOTOS for r in train), "17 Oct val/test photo leaked"
    by_role = defaultdict(list)
    for r in keep:
        by_role[r[0]].append(r)
    paths = Counter(r[1] for r in keep)
    assert all(n == 1 for n in paths.values()), "a crop appears in two files"
    rng = random.Random(s.seed)
    for role in FILES:
        rs = by_role.get(role, [])
        if s.smoke:
            rs = rng.sample(rs, min(len(rs), 40 if role.startswith("train") else 12))
        with (d / f"{role}.txt").open("w", encoding="utf-8") as fh:
            fh.writelines(f"{p}\t{lab}\n" for _, p, lab, _, _ in rs)
        with (d / f"{role}_meta.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["path", "label", "photo", "source"])
            w.writerows((p, lab, ph, src) for _, p, lab, ph, src in rs)
        srcs = Counter(r[4] for r in rs)
        print(f"  {role:<11} {len(rs):>5} lines  {dict(srcs)}"
              + (f"  ({len({r[2] for r in rs})} different texts, {len({r[3] for r in rs})} photos)" if role == "test_b3" else ""))
    if dropped:
        print("  dropped:", dict(dropped))
    print(f"saved label files in {d.relative_to(REPO)}")


# ----------------------------------------------------------------------------------------------- config
def lines_in(path: Path) -> int:
    return sum(1 for _ in path.open(encoding="utf-8"))


def write_config(s: Settings, tag: str = "") -> Path:
    import yaml
    o = run_dir(s)
    o.mkdir(parents=True, exist_ok=True)
    d = data_dir(s)
    cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    try:
        import paddle
        use_gpu = bool(paddle.is_compiled_with_cuda()) and paddle.device.cuda.device_count() > 0
    except Exception:  # noqa: BLE001  (no paddle here: the config is still written)
        use_gpu = False
    bs = s.batch_size
    n_train = s.own_repeat * lines_in(d / "train_own.txt") + int(s.iam_ratio * lines_in(d / "train_iam.txt"))
    steps = max(1, n_train // bs)
    g = cfg["Global"]
    g.update(model_name=MODEL_NAME, use_gpu=use_gpu, distributed=False, epoch_num=s.epochs,
             print_batch_step=1 if s.smoke else 10,                                     # smoke: time per step at once
             save_model_dir=str(o / "checkpoints"), save_epoch_step=s.epochs + 1,        # keep best + latest only
             eval_batch_step=[0, steps],                                                  # evaluate once per epoch
             pretrained_model=str(PRETRAINED), checkpoints=None, character_dict_path=str(DICT),
             max_text_length=s.max_text_length, use_space_char=True, use_visualdl=False,
             save_res_path=str(o / "predicts.txt"), cal_metric_during_train=True)
    cfg["Optimizer"]["lr"].update(learning_rate=s.learning_rate, warmup_epoch=s.warmup_epochs)
    for head in cfg["Architecture"]["Head"]["head_list"]:
        if "NRTRHead" in head:
            head["NRTRHead"]["max_text_length"] = s.max_text_length
    cfg["Metric"]["main_indicator"] = "norm_edit_dis"
    shape, eval_shape = [3, 48, s.train_width], [3, 48, s.eval_width]
    encode = {"MultiLabelEncode": {"gtc_encode": "NRTRLabelEncode"}}
    keys = {"KeepKeys": {"keep_keys": ["image", "label_ctc", "label_gtc", "length", "valid_ratio"]}}
    decode = {"DecodeImage": {"img_mode": "BGR", "channel_first": False}}
    cfg["Train"] = {
        "dataset": {"name": "SimpleDataSet", "data_dir": str(ASSIGNMENT),
                    "label_file_list": [str(d / "train_own.txt")] * s.own_repeat + [str(d / "train_iam.txt")],
                    "ratio_list": [1.0] * s.own_repeat + [s.iam_ratio],
                    "transforms": [decode, {"RecAug": None}, encode, {"RecResizeImg": {"image_shape": shape}}, keys]},
        "loader": {"shuffle": True, "batch_size_per_card": bs, "drop_last": True, "num_workers": s.num_workers,
                   "use_shared_memory": False}}      # rented GPU containers often have a tiny /dev/shm
    cfg["Eval"] = {
        "dataset": {"name": "SimpleDataSet", "data_dir": str(ASSIGNMENT), "label_file_list": [str(d / "val.txt")],
                    "transforms": [decode, encode, {"RecResizeImg": {"image_shape": eval_shape}}, keys]},
        "loader": {"shuffle": False, "drop_last": False, "batch_size_per_card": 32, "num_workers": s.num_workers,
                   "use_shared_memory": False}}
    path = o / f"PP-OCRv6_medium_rec_whiteboard{tag}.yml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    json.dump(asdict(s), (o / "settings.json").open("w"), indent=1)
    print(f"config: {path.relative_to(REPO)} | {'GPU' if use_gpu else 'CPU'} | ~{n_train} training lines per epoch, "
          f"{steps} steps of {bs} | {s.epochs} epochs | train 48 x {s.train_width}, labels <= {s.max_text_length} | "
          f"validation 48 x {s.eval_width} every epoch on {lines_in(d / 'val.txt')} lines (PaddleOCR skips those over "
          f"{s.max_text_length} characters in this check; `evaluate` reads all)")
    return path


# ----------------------------------------------------------------------------------------------- train / export
def run(cmd: list[str], env: dict | None = None) -> None:
    import os
    print("$ " + " ".join(cmd))
    t0 = time.time()
    subprocess.run(cmd, cwd=PADDLEOCR, check=True, env={**os.environ, **(env or {})})
    print(f"  done in {(time.time() - t0) / 60:.1f} min")


def train(s: Settings) -> None:
    if not (PADDLEOCR / "tools" / "train.py").exists():
        sys.exit(f"missing {PADDLEOCR}/tools/train.py (PaddleOCR source checkout)")
    if not PRETRAINED.exists():
        MODELS.mkdir(parents=True, exist_ok=True)
        print(f"downloading {PRETRAINED_URL}")
        urllib.request.urlretrieve(PRETRAINED_URL, PRETRAINED)
    cfg = write_config(s)
    wrapper = s.freeze_bn or s.freeze_backbone
    entry = str(REPO / "scripts" / "paddle_train_frozen_bn.py") if wrapper else "tools/train.py"
    run([sys.executable, entry, "-c", str(cfg)],
        env={"WB_FREEZE_BN": "1" if s.freeze_bn else "0", "WB_FREEZE_BACKBONE": "1" if s.freeze_backbone else "0"})


def export(s: Settings) -> Path:
    o = run_dir(s)
    best = o / "checkpoints" / "best_accuracy"
    if not best.with_suffix(".pdparams").exists():
        sys.exit(f"missing {best}.pdparams: run `train` first")
    target = export_dir(s)
    run([sys.executable, "tools/export_model.py", "-c", str(o / "PP-OCRv6_medium_rec_whiteboard.yml"), "-o",
         f"Global.pretrained_model={best}", f"Global.save_inference_dir={target}"])
    return target


def baseline(s: Settings) -> None:
    """Diagnosis, no training: PaddleOCR's own tools/eval.py with the UNTRAINED pretrained weights on the
    validation lines, under our config, at several image widths. If these already read badly, our input
    pipeline differs from the model's; if they read well but a trained model reads badly, training itself
    (e.g. batch-norm statistics from small batches) does the damage."""
    import copy
    if not PRETRAINED.exists():
        MODELS.mkdir(parents=True, exist_ok=True)
        print(f"downloading {PRETRAINED_URL}")
        urllib.request.urlretrieve(PRETRAINED_URL, PRETRAINED)
    for width in (320, 640):
        s2 = copy.copy(s)
        s2.eval_width = width
        cfg = write_config(s2, tag=f"_eval{width}")
        print(f"\n=== pretrained weights, no training, validation lines resized to 48 x {width} ===")
        run([sys.executable, "tools/eval.py", "-c", str(cfg), "-o", f"Global.pretrained_model={PRETRAINED}"])


# ----------------------------------------------------------------------------------------------- evaluate
def read_all(model, paths, bs=32):
    import cv2
    out = []
    for i in range(0, len(paths), bs):
        imgs = [cv2.imread(str(ASSIGNMENT / p)) for p in paths[i:i + bs]]
        out += [(str(r["rec_text"]), float(r["rec_score"])) for r in model.predict(imgs)]
    return out


def boot_ci(diff_by_photo, n=4000, seed=0):
    rng = random.Random(seed)
    groups = [g for g in diff_by_photo.values() if g]
    if not groups:
        return float("nan"), float("nan")
    means = sorted(sum(sum(g) for g in pick) / sum(len(g) for g in pick)
                   for pick in ([groups[rng.randrange(len(groups))] for _ in groups] for _ in range(n)))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


VAL_NAME = "validation (B2 + 17 Oct val photos)"


def evaluate(s: Settings, model_dir: Path | None = None, val_only: bool = False, header: str = "") -> None:
    """Pretrained vs fine-tuned reader through PaddleOCR's normal reading API, on every line (long ones too).
    val_only: only the validation lines, for choosing between the settings of a comparison (the test board is
    then read once, for the chosen setting only: `select`)."""
    from paddleocr import TextRecognition
    o = run_dir(s)
    model_dir = model_dir or export_dir(s)
    if not model_dir.exists():
        sys.exit(f"missing {model_dir}: run `export` first")
    models = {"pretrained": TextRecognition(model_name=MODEL_NAME),
              "fine-tuned": TextRecognition(model_name=MODEL_NAME, model_dir=str(model_dir))}
    sets = {"Airdrop B3 (unseen board, decides)": "test_b3", "17 Oct test photos": "test_17oct",
            "IAM test (general handwriting)": "test_iam", VAL_NAME: "val"}
    if val_only:
        sets = {VAL_NAME: "val"}
    rows, text, b3, iam = [], [header] if header else [], None, None
    for name, f in sets.items():
        meta = list(csv.DictReader((data_dir(s) / f"{f}_meta.csv").open(encoding="utf-8")))
        if not meta:
            continue
        paths = [m["path"] for m in meta]
        reads = {k: read_all(m, paths) for k, m in models.items()}
        for k in models:
            for m, (t, c) in zip(meta, reads[k]):
                rows.append(dict(set=name, model=k, path=m["path"], photo=m["photo"], label=m["label"], read=t,
                                 confidence=c, cer=cer(m["label"], t)))
        pre = [r for r in rows if r["set"] == name and r["model"] == "pretrained"]
        fin = [r for r in rows if r["set"] == name and r["model"] == "fine-tuned"]
        d = [a["cer"] - b["cer"] for a, b in zip(pre, fin)]                 # + = fine-tuned better
        by_photo = defaultdict(list)
        for r, x in zip(pre, d):
            by_photo[r["photo"]].append(x)
        lo, hi = boot_ci(by_photo)
        mp, mf = sum(r["cer"] for r in pre) / len(pre), sum(r["cer"] for r in fin) / len(fin)
        better, worse = sum(x > 1e-9 for x in d), sum(x < -1e-9 for x in d)
        text.append(f"{name}: {len(pre)} lines | CER pretrained {mp:.3f} -> fine-tuned {mf:.3f} ({mf - mp:+.3f}; gain "
                    f"[95% CI by photo {lo:+.3f}, {hi:+.3f}]) | better {better} / same {len(d) - better - worse} / worse {worse} "
                    f"| exact lines {sum(r['cer'] == 0 for r in pre)} -> {sum(r['cer'] == 0 for r in fin)}")
        if f == "val":
            json.dump(dict(run=s.run, val_cer_pretrained=mp, val_cer_finetuned=mf, better=better, worse=worse,
                           settings=asdict(s)), (o / "val_result.json").open("w"), indent=1)
        if f == "test_b3":
            b3 = (mp, mf, better, worse)
        if f == "test_iam":
            iam = (mp, mf)
    with (o / ("val_comparison.csv" if val_only else "reader_comparison.csv")).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if b3 is not None:
        ok = b3[1] < b3[0] and b3[2] > b3[3] and (iam is None or iam[1] - iam[0] <= 0.02)
        text.append(f"\nVERDICT (rule fixed before training): B3 CER {b3[0]:.3f} -> {b3[1]:.3f}, {b3[2]} better / {b3[3]} worse"
                    + (f"; IAM {iam[0]:.3f} -> {iam[1]:.3f}" if iam is not None else "") + f" -> {'PASSES' if ok else 'FAILS'}")
    examples = [r for r in rows if r["set"].startswith("Airdrop B3")]
    pairs = list(zip(examples[:len(examples) // 2], examples[len(examples) // 2:]))
    changed = [(a, b) for a, b in pairs if a["read"] != b["read"]][:12]
    if changed:
        text.append("\nB3 readings that changed (truth | pretrained | fine-tuned):")
        text += [f"  {a['label']!r} | {a['read']!r} | {b['read']!r}" for a, b in changed]
    summary = "\n".join(text)
    print(summary)
    (o / ("val_summary.txt" if val_only else "summary.txt")).write_text(summary + "\n", encoding="utf-8")


def select(s: Settings, runs: list[str]) -> None:
    """Choose the setting with the lowest VALIDATION CER, copy its exported reader to the app's model folder,
    then read the test board ONCE with it (the decision)."""
    import shutil
    base, rows = base_dir(s), []
    for name in runs:
        f = base / name / "val_result.json"
        if not f.exists():
            print(f"  {name}: no {f.relative_to(REPO)} (did it train / evaluate --val-only?): skipped")
            continue
        rows.append(json.load(f.open()))
    if not rows:
        sys.exit("no run has a validation result")
    win = min(rows, key=lambda r: r["val_cer_finetuned"])
    lines = ["Comparison of fine-tuning settings, chosen on VALIDATION only (lower CER = better):"]
    for r in rows:
        st = r["settings"]
        lines.append(f"  {r['run']:<4} lr {st['learning_rate']:g}, batch {st['batch_size']}, epochs {st['epochs']}, "
                     f"{'backbone frozen' if st['freeze_backbone'] else 'all layers train'}: validation CER "
                     f"{r['val_cer_pretrained']:.3f} (pretrained) -> {r['val_cer_finetuned']:.3f} "
                     f"| better {r['better']} / worse {r['worse']}" + ("   <- chosen" if r is win else ""))
    if win["val_cer_finetuned"] >= win["val_cer_pretrained"]:
        lines.append("  NOTE: no setting beat the pretrained reader on validation; the chosen one is tested anyway (rule).")
    src = base / win["run"] / "inference"
    EXPORT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, EXPORT, dirs_exist_ok=True)
    lines.append(f"  copied {src.relative_to(REPO)} -> {EXPORT.relative_to(REPO)}\n")
    json.dump(dict(chosen=win["run"], runs=rows), (base / "comparison.json").open("w"), indent=1)
    s2 = Settings(**{**asdict(s), "run": ""})
    evaluate(s2, model_dir=EXPORT, header="\n".join(lines))


# ----------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["prepare", "config", "baseline", "train", "export", "evaluate", "select", "all"])
    ap.add_argument("--smoke", action="store_true", help="tiny data, 1 epoch: checks that the whole chain runs")
    for k, v in asdict(Settings()).items():
        if not isinstance(v, bool):
            ap.add_argument(f"--{k.replace('_', '-')}", type=type(v))
    ap.add_argument("--no-freeze-bn", action="store_true", help="let batch-norm update its statistics (PaddleOCR default)")
    ap.add_argument("--freeze-backbone", action="store_true", help="train only the neck and heads")
    ap.add_argument("--val-only", action="store_true", help="evaluate: validation lines only (for a comparison)")
    ap.add_argument("--runs", type=str, default="", help="select: comma-separated run names, e.g. A,B,C")
    a = ap.parse_args()
    s = Settings(smoke=a.smoke)
    if a.smoke:
        s.epochs, s.batch_size, s.own_repeat, s.iam_ratio = 1, 8, 1, 1.0
    for k in asdict(s):
        v = getattr(a, k, None)
        if v is not None and not isinstance(getattr(s, k), bool):
            setattr(s, k, v)
    s.freeze_bn = not a.no_freeze_bn
    s.freeze_backbone = a.freeze_backbone
    if a.step in ("prepare", "all"):
        prepare(s)
    if a.step == "config":
        write_config(s)
    if a.step == "baseline":
        baseline(s)
    if a.step in ("train", "all"):
        train(s)
    if a.step in ("export", "all"):
        export(s)
    if a.step in ("evaluate", "all"):
        evaluate(s, val_only=a.val_only)
    if a.step == "select":
        select(s, [x.strip() for x in a.runs.split(",") if x.strip()])


if __name__ == "__main__":
    main()
