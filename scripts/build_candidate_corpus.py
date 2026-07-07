"""
pipeline.md Section 5 - build the candidate corpus.

Streams `pool_size` documents per language from FineWeb2
(HuggingFaceFW/fineweb-2) via datatrove's ParquetReader, 80% from the
heuristically-clean split and 20% from the `_removed` split by default —
mirroring the pool-building approach in data/pipeline_revise.ipynb, but
without that notebook's length/lang buckets or batch_id (this pilot has no
per-batch human annotation round to stratify for). It keeps that notebook's
bad-word hard-reject (same datatrove `banned_words.txt` asset, same
tokenize-and-intersect check) since that's the one part of "Basic
Prefiltering" this pilot does apply to the candidate corpus -- the
language-score/repetition-score thresholds pipeline.md also lists there are
deliberately *not* applied here; they only gate the `clean_heuristic`
baseline (Section 7 Baseline 2), so the same corpus is what every baseline,
including `clean_heuristic`, starts from.

Also computes Section 5's cheap prefilter features (sea_rater/heuristics.py)
per document -- these need no GPU/rater, so this is the cheapest place to
compute them once, rather than recomputing on every scoring run. Like the
bad-word filter, these are stored as metadata only; they don't remove
anything from the corpus themselves (only `clean_heuristic` acts on them).

Usage:
    python3 scripts/build_candidate_corpus.py --pool-size 100000
    python3 scripts/build_candidate_corpus.py --languages vi --pool-size 1000  # quick test
"""

import argparse
import json
import re
import sys
from pathlib import Path

from datatrove.pipeline.readers import ParquetReader
from datatrove.utils._import_utils import ASSETS_PATH
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sea_rater.heuristics import compute_cheap_features
from sea_rater.languages import LANGUAGE_HF_CONFIGS

HF_DATASET = "HuggingFaceFW/fineweb-2"

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "candidate_corpus"

# Same hard bad-word reject as data/human_annotation/pipeline_revise.ipynb:
# tokenize on non-alphanumeric runs, lowercase, and check for any hit against
# datatrove's built-in banned_words.txt (~400 English adult/profanity terms).
_WORD_SPLITTER = re.compile(r"[^a-zA-Z0-9]+")


def _load_banned_words():
    words = set()
    with open(Path(ASSETS_PATH) / "banned_words.txt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                words.add(_WORD_SPLITTER.sub("", line).lower())
    return words - {""}


BANNED_WORDS = _load_banned_words()


def contains_badword(text):
    tokens = set(_WORD_SPLITTER.split(text.lower()))
    return bool(tokens & BANNED_WORDS)


def stream_docs(hf_config, target_size, source_label):
    path = f"hf://datasets/{HF_DATASET}/data/{hf_config}/train"
    reader = ParquetReader(path)
    rows = []
    progress = tqdm(total=target_size, desc=f"streaming {source_label} [{hf_config}]", unit="doc")
    for doc in reader():
        rows.append(
            {
                "doc_id": doc.id,
                "text": (doc.text or "").strip(),
                "char_len": len(doc.text or ""),
                "language_score": doc.metadata.get("language_score"),
                "source": source_label,
            }
        )
        progress.update(1)
        if len(rows) >= target_size:
            break
    progress.close()
    if len(rows) < target_size:
        print(f"  warning: only found {len(rows)}/{target_size} docs in {path}")
    return rows


def build_language_corpus(lang, hf_config, pool_size, clean_ratio):
    clean_size = round(pool_size * clean_ratio)
    removed_size = pool_size - clean_size

    print(f"[{lang}] streaming {clean_size} clean docs from {hf_config} ...")
    clean_rows = stream_docs(hf_config, clean_size, "clean")

    print(f"[{lang}] streaming {removed_size} removed docs from {hf_config}_removed ...")
    removed_rows = stream_docs(f"{hf_config}_removed", removed_size, "removed")

    rows = clean_rows + removed_rows
    print(f"[{lang}] computing cheap prefilter features for {len(rows)} docs ...")
    for row in tqdm(rows, desc=f"[{lang}] cheap features", unit="doc"):
        row["language"] = lang
        row.update(compute_cheap_features(row["text"], lang))

    print(f"[{lang}] deduplicating by doc_id ...")
    seen = set()
    deduped = []
    for row in rows:
        if row["doc_id"] in seen:
            continue
        seen.add(row["doc_id"])
        deduped.append(row)

    print(f"[{lang}] applying bad-word hard-reject ...")
    cleaned = [row for row in tqdm(deduped, desc=f"[{lang}] bad-word filter", unit="doc") if not contains_badword(row["text"])]

    print(
        f"[{lang}] {len(cleaned)} docs total (clean={len(clean_rows)}, "
        f"removed={len(removed_rows)}, {len(rows) - len(deduped)} duplicate id(s) dropped, "
        f"{len(deduped) - len(cleaned)} bad-word doc(s) dropped)"
    )
    return cleaned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--languages", nargs="+", default=list(LANGUAGE_HF_CONFIGS), choices=list(LANGUAGE_HF_CONFIGS)
    )
    parser.add_argument("--pool-size", type=int, default=100_000)
    parser.add_argument("--clean-ratio", type=float, default=0.8)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for lang in args.languages:
        print(f"\n=== [{lang}] building candidate corpus ===")
        hf_config = LANGUAGE_HF_CONFIGS[lang]
        rows = build_language_corpus(lang, hf_config, args.pool_size, args.clean_ratio)

        out_path = args.output_dir / f"{lang}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{lang}] wrote {len(rows)} docs -> {out_path}\n")


if __name__ == "__main__":
    main()
