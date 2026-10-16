"""
Incrementally sync ./train_data/rec/ to a Hugging Face dataset repo.

- Uploads ONLY new images (name-based diff against the remote repo)
- Always refreshes rec_gt_train.txt and rec_gt_test.txt
- Rebuilds embedded-image train.parquet / test.parquet so
  `load_dataset("user/repo")` works directly
- Writes a README.md with dataset_info frontmatter for the HF viewer
- Single commit per run via a temp staging folder
"""
import os
import shutil
import tempfile
from huggingface_hub import HfApi, create_repo

# ---------- CONFIG ----------
REPO_ID   = "TYW02/iam-line-fork"   # <-- change this
REPO_TYPE = "dataset"
LOCAL_ROOT = "./train_data/rec"
HF_TOKEN  = os.environ.get("HF_TOKEN")      # optional if already logged in

IMAGE_EXTS   = (".jpg", ".jpeg", ".png", ".bmp")
LABEL_FILES  = ["rec_gt_train.txt", "rec_gt_test.txt"]
SPLITS = {
    "train": ("train.parquet", "train"),
    "test":  ("test.parquet",  "test"),
}
BUILD_PARQUET = True   # set False to skip parquet rebuild (fast label-only sync)
# ----------------------------

api = HfApi()
create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, exist_ok=True, token=HF_TOKEN)

# ------------------------------------------------------------------
# 1. List existing files in the repo
# ------------------------------------------------------------------
print(f"Listing files in {REPO_ID} ...")
existing_files = set(
    api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE, token=HF_TOKEN)
)
print(f"  {len(existing_files)} files already present in the repo")


# ------------------------------------------------------------------
# 2. Parse label files into {repo_relative_path: label}
# ------------------------------------------------------------------
def parse_label_file(path):
    mapping = {}
    if not os.path.isfile(path):
        print(f"  WARNING: label file not found: {path}")
        return mapping
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                print(f"  WARNING: {path}:{i} malformed (no tab): {line!r}")
                continue
            img_rel, label = parts
            mapping[img_rel] = label
    return mapping


train_labels = parse_label_file(os.path.join(LOCAL_ROOT, "rec_gt_train.txt"))
test_labels  = parse_label_file(os.path.join(LOCAL_ROOT, "rec_gt_test.txt"))
print(f"Parsed {len(train_labels)} train labels, {len(test_labels)} test labels")

split_labels = {"train": train_labels, "test": test_labels}


# ------------------------------------------------------------------
# 3. Build parquet files with embedded images
# ------------------------------------------------------------------
def build_parquet(split_name, local_img_dir, label_map, out_path):
    try:
        from datasets import Dataset, Features, Image as HFImage, Value
    except ImportError:
        print("ERROR: `datasets` is not installed. Run: pip install datasets")
        raise

    paths, texts = [], []
    skipped = 0
    for fname in sorted(os.listdir(local_img_dir)):
        if not fname.lower().endswith(IMAGE_EXTS):
            continue
        repo_rel = f"{split_name}/{fname}"
        if repo_rel not in label_map:
            skipped += 1
            continue
        local_path = os.path.join(local_img_dir, fname)
        if not os.path.isfile(local_path):
            skipped += 1
            continue
        paths.append(local_path)
        texts.append(label_map[repo_rel])

    if not paths:
        print(f"  {out_path}: no usable images, skipping parquet")
        return 0

    features = Features({
        "image": HFImage(),
        "text":  Value("string"),
    })
    ds = Dataset.from_dict({"image": paths, "text": texts}, features=features)
    ds.to_parquet(out_path)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  {out_path}: {len(paths)} rows, {size_mb:.1f} MB "
          f"({skipped} images skipped: no label)")
    return len(paths)


# ------------------------------------------------------------------
# 4. Stage everything into a temp folder
# ------------------------------------------------------------------
new_images      = 0
skipped_images  = 0
label_counts    = {}
parquet_counts  = {}

with tempfile.TemporaryDirectory() as staging_dir:
    # 4a. Stage new images only (diff)
    for split_name in SPLITS:
        local_sub = os.path.join(LOCAL_ROOT, split_name)
        if not os.path.isdir(local_sub):
            print(f"  Skipping missing folder: {local_sub}")
            continue

        for fname in sorted(os.listdir(local_sub)):
            local_path = os.path.join(local_sub, fname)
            if not os.path.isfile(local_path):
                continue
            if not fname.lower().endswith(IMAGE_EXTS):
                continue

            repo_path = f"{split_name}/{fname}"
            if repo_path in existing_files:
                skipped_images += 1
                continue

            dest = os.path.join(staging_dir, split_name, fname)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(local_path, dest)
            new_images += 1

    # 4b. Always refresh label files
    for label in LABEL_FILES:
        src = os.path.join(LOCAL_ROOT, label)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(staging_dir, label))
            label_counts[label] = "staged"
        else:
            print(f"  WARNING: label file not found: {src}")

    # 4c. Build parquet from the FULL local dataset
    if BUILD_PARQUET:
        print("\nBuilding parquet files (full local dataset) ...")
        for split_name, (parquet_name, _) in SPLITS.items():
            local_sub = os.path.join(LOCAL_ROOT, split_name)
            if not os.path.isdir(local_sub):
                continue
            out_path = os.path.join(staging_dir, parquet_name)
            n = build_parquet(
                split_name,
                local_sub,
                split_labels[split_name],
                out_path,
            )
            if n:
                parquet_counts[parquet_name] = n

    # 4d. Write README.md
    num_train = parquet_counts.get("train.parquet", len(train_labels))
    num_test  = parquet_counts.get("test.parquet",  len(test_labels))

    fence = "`" * 3   # avoid literal triple backticks in source

    readme = (
        "---\n"
        "license: other\n"
        "task_categories:\n"
        "- image-to-text\n"
        "language:\n"
        "- en\n"
        "tags:\n"
        "- handwriting\n"
        "- ocr\n"
        "configs:\n"
        "- config_name: default\n"
        "  data_files:\n"
        "  - split: train\n"
        "    path: train.parquet\n"
        "  - split: test\n"
        "    path: test.parquet\n"
        "dataset_info:\n"
        "  features:\n"
        "  - name: image\n"
        "    dtype: image\n"
        "  - name: text\n"
        "    dtype: string\n"
        "  splits:\n"
        "  - name: train\n"
        f"    num_examples: {num_train}\n"
        "  - name: test\n"
        f"    num_examples: {num_test}\n"
        "---\n"
        "\n"
        "# IAM-line fork\n"
        "\n"
        "Line-level handwritten text dataset for PaddleOCR fine-tuning.\n"
        "\n"
        "Layout:\n"
        "- `train/`, `test/` — loose image files\n"
        "- `train.parquet`, `test.parquet` — same data, images embedded, "
        "loadable via `load_dataset()`\n"
        "- `rec_gt_train.txt`, `rec_gt_test.txt` — PaddleOCR-format labels "
        "(`path\\ttext`)\n"
        "\n"
        f"{fence}python\n"
        "from datasets import load_dataset\n"
        f'ds = load_dataset("{REPO_ID}")\n'
        'print(ds["train"][0])\n'
        f"{fence}\n"
    )

    with open(os.path.join(staging_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    # 4e. Summary
    print("\n" + "=" * 55)
    print(f"  New images staged : {new_images}")
    print(f"  Skipped (existing): {skipped_images}")
    print(f"  Label files       : {list(label_counts.keys())}")
    print(f"  Parquet files     : {parquet_counts}")
    print("=" * 55 + "\n")

    if new_images == 0 and not parquet_counts and not label_counts:
        print("Nothing to upload. Exiting.")
        raise SystemExit(0)

    # 4f. One commit, one upload
    print(f"Uploading to {REPO_ID} ...")
    api.upload_folder(
        folder_path=staging_dir,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        token=HF_TOKEN,
        commit_message=(
            f"Sync: +{new_images} images, "
            f"parquet={list(parquet_counts.keys())}"
        ),
    )

print("\nDone.")