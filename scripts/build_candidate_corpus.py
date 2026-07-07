"""
pipeline.md Section 5 - build the candidate corpus.

Streams `pool_size` documents per language from FineWeb2
(HuggingFaceFW/fineweb-2) via datatrove's ParquetReader, 80% from the
heuristically-clean split and 20% from the `_removed` split by default —
mirroring the pool-building approach in data/pipeline_revise.ipynb, but
without that notebook's adult-content filter, length filter, length/lang
buckets, or batch_id (pipeline.md explicitly leaves that prefiltering
unimplemented for this pilot; Section 5 only asks for
doc_id/language/text/source).

Usage:
    python3 scripts/build_candidate_corpus.py --pool-size 250000
    python3 scripts/build_candidate_corpus.py --languages vi --pool-size 1000  # quick test
"""

import argparse
import json
from pathlib import Path

from datatrove.pipeline.readers import ParquetReader

HF_DATASET = "HuggingFaceFW/fineweb-2"
LANGUAGE_HF_CONFIGS = {
    "vi": "vie_Latn",
    "id": "ind_Latn",
    "th": "tha_Thai",
    "km": "khm_Khmr",
}

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "candidate_corpus"


def stream_docs(hf_config, target_size, source_label):
    path = f"hf://datasets/{HF_DATASET}/data/{hf_config}/train"
    reader = ParquetReader(path)
    rows = []
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
        if len(rows) >= target_size:
            break
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
    for row in rows:
        row["language"] = lang

    seen = set()
    deduped = []
    for row in rows:
        if row["doc_id"] in seen:
            continue
        seen.add(row["doc_id"])
        deduped.append(row)

    print(
        f"[{lang}] {len(deduped)} docs total (clean={len(clean_rows)}, "
        f"removed={len(removed_rows)}, {len(rows) - len(deduped)} duplicate id(s) dropped)"
    )
    return deduped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--languages", nargs="+", default=list(LANGUAGE_HF_CONFIGS), choices=list(LANGUAGE_HF_CONFIGS)
    )
    parser.add_argument("--pool-size", type=int, default=250_000)
    parser.add_argument("--clean-ratio", type=float, default=0.8)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for lang in args.languages:
        hf_config = LANGUAGE_HF_CONFIGS[lang]
        rows = build_language_corpus(lang, hf_config, args.pool_size, args.clean_ratio)

        out_path = args.output_dir / f"{lang}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{lang}] wrote {len(rows)} docs -> {out_path}\n")


if __name__ == "__main__":
    main()
