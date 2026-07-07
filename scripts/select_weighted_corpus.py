"""
pipeline.md Section 7 - Main method: Best weighted combination.

For each of the 26 weight combinations (Section 8, data/weight_combinations.json)
and each language, computes a weighted aggregate score per scored candidate
document:

    score = w1*educational_value + w2*reasoning + w3*professionalism
          + w4*cleanliness + w5*cultural_nuance

ranks documents by that score, and selects the top documents until a
per-language token budget is met -- this is "Data selected by the weight
combination" that Section 9's proxy CPT trains on, one dataset per
(weight_id, language) pair.

Token counts are estimated as char_len / chars_per_token (default 4), a
cheap heuristic rather than running a real tokenizer, since this only
needs to decide how many documents to include, not exact training counts.

Usage:
    python3 scripts/select_weighted_corpus.py
    python3 scripts/select_weighted_corpus.py --weight-ids W01 W06 --languages vi
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sea_rater.languages import LANGUAGES
from sea_rater.selection import CORPUS_DIR, join_scores_with_text, select_for_weight

DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "data" / "weight_combinations.json"
DEFAULT_OUTPUT_DIR = CORPUS_DIR / "selected"

DEFAULT_TOKENS_PER_LANGUAGE = 2_000_000  # pipeline.md Section 9 proxy data size


def load_weight_combinations(path, weight_ids=None):
    combos = json.loads(path.read_text(encoding="utf-8"))
    if weight_ids:
        combos = [c for c in combos if c["id"] in weight_ids]
    return combos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight-ids", nargs="+", default=None, help="subset of weight ids, e.g. W01 W06")
    parser.add_argument("--languages", nargs="+", default=LANGUAGES, choices=LANGUAGES)
    parser.add_argument("--weights-path", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--tokens-per-language", type=int, default=DEFAULT_TOKENS_PER_LANGUAGE)
    parser.add_argument("--chars-per-token", type=float, default=4.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    combos = load_weight_combinations(args.weights_path, args.weight_ids)
    print(f"Loaded {len(combos)} weight combination(s) from {args.weights_path}")

    joined_by_lang = {}
    for lang in args.languages:
        print(f"[{lang}] joining scored corpus with text ...")
        joined_by_lang[lang] = join_scores_with_text(lang)
        print(f"[{lang}] {len(joined_by_lang[lang])} scored docs available")

    for combo in tqdm(combos, desc="weight combinations", unit="combo"):
        weight_id, weights = combo["id"], combo["weights"]
        out_dir = args.output_dir / weight_id
        out_dir.mkdir(parents=True, exist_ok=True)

        for lang in args.languages:
            docs = [dict(d) for d in joined_by_lang[lang]]  # per-combo copy, since we mutate in place
            selected, total_tokens = select_for_weight(
                docs, weights, args.tokens_per_language, args.chars_per_token
            )

            out_path = out_dir / f"{lang}.jsonl"
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

            tqdm.write(
                f"[{weight_id}][{lang}] selected {len(selected)} docs, "
                f"~{total_tokens / 1e6:.2f}M est. tokens "
                f"(target {args.tokens_per_language / 1e6:.2f}M) -> {out_path}"
            )


if __name__ == "__main__":
    main()
