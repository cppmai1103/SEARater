"""
pipeline.md Section 3 - Human Annotation Data.

Reads the human-labeled CSVs for the 4 pilot languages and splits each
language's 940 documents into train/dev/test/drop (720/90/90/40), using
iterative stratification so the 0-5 score distribution of every one of the
5 quality dimensions stays proportionally balanced across splits. Language
balance is exact by construction, since each language is split
independently with the same fixed target sizes.

Usage:
    python3 scripts/split_human_annotations.py
"""

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(sys.maxsize)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = DATA_DIR / "splits"

LANGUAGE_FILES = {
    "vi": "/human_annotation/vie_Latn.csv",
    "id": "/human_annotation/ind_Latn.csv",
    "th": "/human_annotation/tha_Thai.csv",
    "km": "/human_annotation/khm_Khmr.csv",
}

DIMENSIONS = [
    "educational_value",
    "reasoning",
    "professionalism",
    "cleanliness",
    "cultural_nuances",
]

# 940 available per language; 900 of them (720/90/90) go to train/dev/test,
# the remaining 40 are dropped. Sizes must sum to the row count per language.
SPLIT_SIZES = {"train": 720, "dev": 90, "test": 90, "drop": 40}
SEED = 42


def row_labels(row):
    """One (dimension, score) label per quality dimension for this row."""
    return [(dim, row[dim]) for dim in DIMENSIONS]


def load_rows(csv_path, lang):
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for row in rows:
        row["language"] = lang
    return rows


def iterative_stratified_split(rows, label_fn, split_sizes, rng):
    """Assign each row to one of `split_sizes` (name -> exact target count),
    balancing the marginal distribution of every label returned by
    `label_fn` across splits (Sechidis et al., "On the Stratification of
    Multi-Label Data", 2011).

    Labels here are (dimension, score) pairs, so this balances the 0-5
    histogram of every quality dimension simultaneously, without needing a
    combinatorial join key over all 5 dimensions at once.
    """
    total = len(rows)
    assert sum(split_sizes.values()) == total, (
        f"split sizes {split_sizes} do not sum to {total} rows"
    )
    names = list(split_sizes.keys())
    ratios = {name: size / total for name, size in split_sizes.items()}

    labels_of = [label_fn(row) for row in rows]
    label_rows = defaultdict(set)
    for i, labels in enumerate(labels_of):
        for label in labels:
            label_rows[label].add(i)

    # Desired remaining count of each label per split, proportional to split size.
    desired = {
        label: {name: ratios[name] * len(idxs) for name in names}
        for label, idxs in label_rows.items()
    }
    remaining_target = dict(split_sizes)
    unassigned = set(range(total))
    assignment = {}

    while unassigned:
        active = {label: idxs & unassigned for label, idxs in label_rows.items()}
        active = {label: idxs for label, idxs in active.items() if idxs}
        if not active:
            # Rows with no labels left (shouldn't happen: every row has 5
            # dimension labels) - assign arbitrarily to whichever split has room.
            for i in list(unassigned):
                name = max(names, key=lambda n: remaining_target[n])
                assignment[i] = name
                remaining_target[name] -= 1
                unassigned.discard(i)
            break

        # Process the rarest label first (fewest remaining candidate rows).
        label = min(active, key=lambda l: len(active[l]))
        candidate_idxs = list(active[label])
        rng.shuffle(candidate_idxs)

        for i in candidate_idxs:
            open_names = [n for n in names if remaining_target[n] > 0]
            best = max(
                open_names,
                key=lambda n: (desired[label][n], remaining_target[n], rng.random()),
            )
            assignment[i] = best
            remaining_target[best] -= 1
            unassigned.discard(i)
            for l in labels_of[i]:
                desired[l][best] -= 1

    result = {name: [] for name in names}
    for i, name in assignment.items():
        result[name].append(rows[i])
    return result


def to_record(row):
    raw_id = row["id"].strip("<>").removeprefix("urn:uuid:")
    record = {
        "doc_id": f"{row['language']}_{raw_id}",
        "language": row["language"],
        "text": row["text"],
        "educational_value": int(row["educational_value"]),
        "reasoning": int(row["reasoning"]),
        "professionalism": int(row["professionalism"]),
        "cleanliness": int(row["cleanliness"]),
        "cultural_nuance": int(row["cultural_nuances"]),
        "source": row["source"],
        "len_bucket": row["len_bucket"],
        "lang_bucket": row["lang_bucket"],
        "char_len": int(row["char_len"]),
    }
    return record


def report(name, rows):
    by_lang = defaultdict(int)
    for row in rows:
        by_lang[row["language"]] += 1
    print(f"\n{name}: {len(rows)} docs")
    print("  by language:", dict(sorted(by_lang.items())))
    for dim in DIMENSIONS:
        by_score = defaultdict(int)
        for row in rows:
            by_score[row[dim]] += 1
        print(f"  by {dim}:", dict(sorted(by_score.items())))


def main():
    rng = random.Random(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = {"train": [], "dev": [], "test": []}

    for lang, filename in LANGUAGE_FILES.items():
        csv_path = DATA_DIR / filename
        rows = load_rows(csv_path, lang)
        assert len(rows) == sum(SPLIT_SIZES.values()), (
            f"{filename} has {len(rows)} rows, expected {sum(SPLIT_SIZES.values())}"
        )

        lang_splits = iterative_stratified_split(rows, row_labels, SPLIT_SIZES, rng)

        for name in splits:
            splits[name].extend(lang_splits[name])
        # lang_splits["drop"] is intentionally discarded.

    for name, rows in splits.items():
        rng.shuffle(rows)
        out_path = OUTPUT_DIR / f"human_{name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(to_record(row), ensure_ascii=False) + "\n")
        report(f"human_{name}.jsonl", rows)

    print(f"\nWrote splits to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
