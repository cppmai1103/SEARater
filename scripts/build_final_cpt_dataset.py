"""
pipeline.md Section 11 - Final Continued Pretraining Datasets.

Builds the final 50M-token (12.5M/language) training set for
Qwen2.5-1.5B-CPT-BestWeighted, using the single winning weight vector from
Section 10 (data/best_weight.json) applied to the full scored candidate
corpus -- same ranking algorithm as the proxy selection
(scripts/select_weighted_corpus.py, Section 7), just one weight vector
instead of 16, and a much larger per-language token budget.

The pilot's other final "dataset" -- the original, non-continued-pretrained
Qwen2.5-1.5B Base -- needs no data at all, so there's nothing to build for
it here; it's just evaluated as-is in Section 13.

Usage:
    python3 scripts/build_final_cpt_dataset.py
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sea_rater.selection import CORPUS_DIR, join_scores_with_text, select_for_weight

DEFAULT_BEST_WEIGHT_PATH = Path(__file__).resolve().parent.parent / "data" / "best_weight.json"
DEFAULT_OUTPUT_DIR = CORPUS_DIR / "final_cpt_dataset"

LANGUAGES = ["vi", "id", "th", "km"]
DEFAULT_TOKENS_PER_LANGUAGE = 12_500_000  # pipeline.md Section 11 final data size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="+", default=LANGUAGES, choices=LANGUAGES)
    parser.add_argument("--best-weight-path", type=Path, default=DEFAULT_BEST_WEIGHT_PATH)
    parser.add_argument("--tokens-per-language", type=int, default=DEFAULT_TOKENS_PER_LANGUAGE)
    parser.add_argument("--chars-per-token", type=float, default=4.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    best = json.loads(args.best_weight_path.read_text(encoding="utf-8"))
    weights = best["weights"]
    print(f"Using best weight {best['weight_id']} ({best['meaning']}): {weights}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for lang in args.languages:
        print(f"[{lang}] joining scored corpus with text ...")
        docs = join_scores_with_text(lang)
        print(f"[{lang}] {len(docs)} scored docs available")

        selected, total_tokens = select_for_weight(
            docs, weights, args.tokens_per_language, args.chars_per_token
        )

        out_path = args.output_dir / f"{lang}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for doc in selected:
                record = {
                    "doc_id": doc["doc_id"],
                    "language": doc["language"],
                    "source": doc.get("source"),
                    "weighted_score": doc["weighted_score"],
                    "estimated_tokens": doc["estimated_tokens"],
                    "text": doc["text"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(
            f"[{lang}] selected {len(selected)} docs, ~{total_tokens / 1e6:.2f}M est. tokens "
            f"(target {args.tokens_per_language / 1e6:.2f}M) -> {out_path}"
        )


if __name__ == "__main__":
    main()
