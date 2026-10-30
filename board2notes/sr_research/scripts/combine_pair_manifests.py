#!/usr/bin/env python3
"""Combine frontal and side-angle candidate manifests into one review queue."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "outputs" / "paired_dataset_17oct"
SOURCES = [
    ("frontal", REPO / "outputs" / "paired_dataset_17oct_full"),
    ("side", REPO / "outputs" / "paired_dataset_17oct_side"),
]

REVIEW_FIELDS = [
    "transcription_verified",
    "review_notes",
    "cer_eligible",
    "assistant_decision",
    "annotation_source",
]


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_or_remove(path: Path, rows: list[dict]) -> None:
    """Write a derived manifest, or remove an obsolete copy when it is empty."""
    if rows:
        write_rows(path, rows)
    elif path.exists():
        path.unlink()


def main() -> None:
    previous = {}
    existing = OUTPUT / "candidate_manifest.csv"
    if existing.exists():
        previous = {row["pair_id"]: row for row in read_rows(existing)}
    combined: list[dict] = []
    seen: set[str] = set()
    for source_set, folder in SOURCES:
        for row in read_rows(folder / "manifest.csv"):
            # Synthetic samples are generated from the same 20 close targets in
            # both runs. Keep the frontal copy once; real camera pairs are unique.
            if row["pair_id"] in seen:
                continue
            seen.add(row["pair_id"])
            row = dict(row)
            row["source_set"] = source_set
            row["input_path"] = (folder.relative_to(REPO) / row["input_path"]).as_posix()
            row["target_path"] = (folder.relative_to(REPO) / row["target_path"]).as_posix()
            old = previous.get(row["pair_id"], {})
            for field in REVIEW_FIELDS:
                row[field] = old.get(field, "")
            if old.get("review_status") in {"accepted", "rejected"}:
                row["review_status"] = old["review_status"]
            combined.append(row)

    # Put the pairs most likely to need attention first.
    def priority(row: dict) -> tuple:
        return (
            row["pair_type"] != "real",
            row.get("geometry_review_priority") != "high",
            row.get("transcription_review_priority") != "high",
            row.get("source_set") != "side",
            row["input_photo"],
            row["pair_id"],
        )

    review_queue = sorted((r for r in combined if r["pair_type"] == "real"), key=priority)
    write_rows(OUTPUT / "candidate_manifest.csv", combined)
    write_rows(OUTPUT / "review_queue.csv", review_queue)

    accepted = [r for r in combined if r["review_status"] == "accepted"]
    write_or_remove(OUTPUT / "accepted_manifest.csv", accepted)

    rejected = [r for r in combined if r["review_status"] == "rejected"]
    write_or_remove(OUTPUT / "rejected_manifest.csv", rejected)

    needs_user_review = [r for r in combined if r["review_status"] == "needs_review"]
    write_or_remove(OUTPUT / "needs_user_review.csv", needs_user_review)

    counts = Counter((r["pair_type"], r["source_set"]) for r in combined)
    geometry = Counter(r["geometry_review_priority"] for r in review_queue)
    text = Counter(r["transcription_review_priority"] for r in review_queue)
    annotation_note = (
        "Review status is based on super-resolution pair quality: the two crops must contain "
        "the same writing, be sufficiently aligned, and have a genuinely clearer target. "
        "Transcriptions are optional metadata and are not required for enhancement training."
    )
    summary = f"""# 17 October paired-dataset preparation summary

- Unique candidate pairs: **{len(combined)}**
- Real frontal pairs: **{counts[('real', 'frontal')]}**
- Real side-angle pairs: **{counts[('real', 'side')]}**
- Unique synthetic pairs: **{sum(v for (kind, _), v in counts.items() if kind == 'synthetic')}**
- Real pairs with normal geometry priority: **{geometry['normal']}**
- Real pairs with high geometry review priority: **{geometry['high']}**
- Real pairs with high transcription review priority: **{text['high']}**
- Accepted pairs: **{len(accepted)}**
- Rejected pairs: **{len(rejected)}**
- Pairs still needing visual review: **{len(needs_user_review)}**

All items belong to `17oct_board01` and must remain in one final data split.
{annotation_note}

Files:

- `candidate_manifest.csv`: all real and de-duplicated synthetic candidates.
- `review_queue.csv`: real pairs only, with high-priority cases first.
- `accepted_manifest.csv`: pairs approved for enhancement training.
- `rejected_manifest.csv`: unusable or mismatched pairs.
- `needs_user_review.csv`: unresolved pairs, when any remain.
"""
    (OUTPUT / "README.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
