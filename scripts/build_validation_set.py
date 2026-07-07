"""
pipeline.md Section 9 - build the fixed proxy/final validation set.

Streams a small, fixed held-out set per language (1M tokens/language, 8M
total) from the *same* FineWeb2 splits build_candidate_corpus.py draws
from, but skipping past however many documents that script already
consumed -- so this validation set never overlaps with candidate training
data, without needing to dedupe against a huge in-memory ID set.

Token counts are estimated as char_len / chars_per_token, same cheap
heuristic used in select_weighted_corpus.py.

Usage:
    python3 scripts/build_validation_set.py
"""

import argparse
import json
import sys
from pathlib import Path

from datatrove.pipeline.readers import ParquetReader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sea_rater.languages import LANGUAGE_HF_CONFIGS

HF_DATASET = "HuggingFaceFW/fineweb-2"

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "validation_set"

# Must match build_candidate_corpus.py's actual --pool-size / --clean-ratio,
# so the skip counts below land exactly past what that script already read.
CANDIDATE_POOL_SIZE = 100_000
CANDIDATE_CLEAN_RATIO = 0.8
SKIP_CLEAN = round(CANDIDATE_POOL_SIZE * CANDIDATE_CLEAN_RATIO)
SKIP_REMOVED = CANDIDATE_POOL_SIZE - SKIP_CLEAN

DEFAULT_TOKENS_PER_LANGUAGE = 1_000_000
DEFAULT_CLEAN_RATIO = 0.8


def stream_docs_after_skip(hf_config, skip_n, token_target, source_label, chars_per_token):
    path = f"hf://datasets/{HF_DATASET}/data/{hf_config}/train"
    reader = ParquetReader(path)
    rows = []
    total_tokens = 0
    skipped = 0
    skip_progress = tqdm(total=skip_n, desc=f"skipping already-consumed [{source_label}]", unit="doc")
    token_progress = tqdm(total=token_target, desc=f"streaming {source_label} [{hf_config}]", unit="tok")
    for doc in reader():
        if skipped < skip_n:
            skipped += 1
            skip_progress.update(1)
            continue
        text = (doc.text or "").strip()
        char_len = len(text)
        rows.append(
            {
                "doc_id": doc.id,
                "text": text,
                "char_len": char_len,
                "source": source_label,
            }
        )
        est_tokens = char_len / chars_per_token
        total_tokens += est_tokens
        token_progress.update(min(est_tokens, token_target - token_progress.n))
        if total_tokens >= token_target:
            break
    skip_progress.close()
    token_progress.close()
    if total_tokens < token_target:
        print(
            f"  warning: only found ~{total_tokens / 1e6:.2f}M tokens in {path} "
            f"after skipping {skip_n}, wanted {token_target / 1e6:.2f}M"
        )
    return rows, total_tokens


def build_language_validation_set(lang, hf_config, tokens_target, clean_ratio, chars_per_token):
    clean_tokens = tokens_target * clean_ratio
    removed_tokens = tokens_target - clean_tokens

    print(f"[{lang}] streaming ~{clean_tokens / 1e6:.2f}M clean tokens (skip {SKIP_CLEAN}) from {hf_config} ...")
    clean_rows, clean_total = stream_docs_after_skip(
        hf_config, SKIP_CLEAN, clean_tokens, "clean", chars_per_token
    )

    print(
        f"[{lang}] streaming ~{removed_tokens / 1e6:.2f}M removed tokens (skip {SKIP_REMOVED}) "
        f"from {hf_config}_removed ..."
    )
    removed_rows, removed_total = stream_docs_after_skip(
        f"{hf_config}_removed", SKIP_REMOVED, removed_tokens, "removed", chars_per_token
    )

    rows = clean_rows + removed_rows
    for row in rows:
        row["language"] = lang

    total_tokens = clean_total + removed_total
    print(
        f"[{lang}] {len(rows)} validation docs, ~{total_tokens / 1e6:.2f}M est. tokens "
        f"(clean={len(clean_rows)}, removed={len(removed_rows)})"
    )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="+", default=list(LANGUAGE_HF_CONFIGS), choices=list(LANGUAGE_HF_CONFIGS))
    parser.add_argument("--tokens-per-language", type=int, default=DEFAULT_TOKENS_PER_LANGUAGE)
    parser.add_argument("--clean-ratio", type=float, default=DEFAULT_CLEAN_RATIO)
    parser.add_argument("--chars-per-token", type=float, default=4.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for lang in args.languages:
        print(f"\n=== [{lang}] building validation set ===")
        hf_config = LANGUAGE_HF_CONFIGS[lang]
        rows = build_language_validation_set(
            lang, hf_config, args.tokens_per_language, args.clean_ratio, args.chars_per_token
        )

        out_path = args.output_dir / f"{lang}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{lang}] wrote {len(rows)} docs -> {out_path}\n")


if __name__ == "__main__":
    main()
