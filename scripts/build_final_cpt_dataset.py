"""
pipeline.md Section 11 - Final Continued Pretraining Datasets.

Builds the final 50M-token (12.5M/language) training set for one of
Section 7's final baselines -- random, clean-only heuristic, equal-average,
or the pilot's main method, best-weighted (default) -- using
the same ranking/filtering logic (sea_rater/selection.py) applied to the
full scored candidate corpus, just a much larger per-language token budget
than the proxy selection (scripts/select_weighted_corpus.py, Section 7)
used.

The pilot's other final "dataset" -- the original, non-continued-pretrained
Qwen2.5-1.5B Base -- needs no data at all, so there's nothing to build for
it here; it's just evaluated as-is in Section 13.

Usage:
    python3 scripts/build_final_cpt_dataset.py                        # best_weighted (default)
    python3 scripts/build_final_cpt_dataset.py --baseline random
    python3 scripts/build_final_cpt_dataset.py --baseline clean_heuristic
    python3 scripts/build_final_cpt_dataset.py --baseline equal_average
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sea_rater.baselines import BASELINE_CHOICES, DATASET_DIR_BY_BASELINE
from sea_rater.languages import LANGUAGES
from sea_rater.selection import join_scores_with_text, select_for_baseline

DEFAULT_BEST_WEIGHT_PATH = Path(__file__).resolve().parent.parent / "data" / "best_weight.json"

DEFAULT_TOKENS_PER_LANGUAGE = 12_500_000  # pipeline.md Section 11 final data size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=BASELINE_CHOICES, default="best_weighted")
    parser.add_argument("--languages", nargs="+", default=LANGUAGES, choices=LANGUAGES)
    parser.add_argument("--best-weight-path", type=Path, default=DEFAULT_BEST_WEIGHT_PATH)
    parser.add_argument("--tokens-per-language", type=int, default=DEFAULT_TOKENS_PER_LANGUAGE)
    parser.add_argument("--chars-per-token", type=float, default=4.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or DATASET_DIR_BY_BASELINE[args.baseline]

    best_weights = None
    if args.baseline == "best_weighted":
        best = json.loads(args.best_weight_path.read_text(encoding="utf-8"))
        best_weights = best["weights"]
        print(f"Using best weight {best['weight_id']} ({best['meaning']}): {best_weights}")
    else:
        print(f"Building final CPT dataset for baseline '{args.baseline}'")

    output_dir.mkdir(parents=True, exist_ok=True)

    for lang in tqdm(args.languages, desc=f"building '{args.baseline}' dataset", unit="lang"):
        tqdm.write(f"[{lang}] joining scored corpus with text ...")
        docs = join_scores_with_text(lang)
        tqdm.write(f"[{lang}] {len(docs)} scored docs available")

        selected, total_tokens = select_for_baseline(
            args.baseline, docs, args.tokens_per_language, args.chars_per_token, best_weights=best_weights
        )

        out_path = output_dir / f"{lang}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for doc in selected:
                record = {
                    "doc_id": doc["doc_id"],
                    "language": doc["language"],
                    "source": doc.get("source"),
                    "weighted_score": doc.get("weighted_score"),
                    "estimated_tokens": doc["estimated_tokens"],
                    "text": doc["text"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        tqdm.write(
            f"[{lang}] selected {len(selected)} docs, ~{total_tokens / 1e6:.2f}M est. tokens "
            f"(target {args.tokens_per_language / 1e6:.2f}M) -> {out_path}"
        )
        if len(selected) == 0:
            tqdm.write(
                f"  warning: [{lang}] selected 0 docs for baseline '{args.baseline}' -- "
                f"downstream packing will crash on an empty text list. If this is "
                f"'clean_heuristic', the likely cause is that data/candidate_corpus/{lang}.jsonl "
                f"predates sea_rater/heuristics.py's cheap-feature fields (repetition_score, "
                f"target_script_ratio, ...); rerun build_candidate_corpus.py for this language "
                f"to backfill them."
            )
        elif total_tokens < args.tokens_per_language * 0.5:
            tqdm.write(
                f"  warning: [{lang}] only reached {total_tokens / 1e6:.2f}M of the "
                f"{args.tokens_per_language / 1e6:.2f}M-token target (less than half) -- the "
                f"candidate/scored corpus for this language may be too small or too heavily "
                f"filtered for baseline '{args.baseline}'."
            )


if __name__ == "__main__":
    main()
