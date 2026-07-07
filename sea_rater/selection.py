"""
Shared weighted-selection logic, used by both the proxy selection
(scripts/select_weighted_corpus.py, Section 7) and the final CPT dataset
build (scripts/build_final_cpt_dataset.py, Section 11) -- same ranking
algorithm, different weight source (one of 16 fixed combos vs. the single
best_weight.json winner) and different token budget (2M/language proxy vs.
12.5M/language final).
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "data" / "candidate_corpus"
SCORED_DIR = CORPUS_DIR / "scored"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def join_scores_with_text(lang):
    """Join scored_corpus/{lang}.jsonl (doc_id + 5 scores) back to its text
    in candidate_corpus/{lang}.jsonl by doc_id."""
    scored_rows = load_jsonl(SCORED_DIR / f"{lang}.jsonl")
    corpus_rows = load_jsonl(CORPUS_DIR / f"{lang}.jsonl")
    text_by_id = {row["doc_id"]: row for row in corpus_rows}

    joined = []
    missing = 0
    for row in scored_rows:
        corpus_row = text_by_id.get(row["doc_id"])
        if corpus_row is None:
            missing += 1
            continue
        joined.append({**row, "text": corpus_row["text"], "char_len": corpus_row["char_len"]})
    if missing:
        print(f"  warning: {missing} scored doc(s) for '{lang}' had no matching corpus text, skipped")
    return joined


def select_for_weight(docs, weights, tokens_target, chars_per_token):
    """Rank `docs` by their weighted score and keep the top ones until the
    estimated token budget (char_len / chars_per_token) is met."""
    for doc in docs:
        doc["weighted_score"] = sum(weights[dim] * doc[dim] for dim in weights)
    ranked = sorted(docs, key=lambda d: d["weighted_score"], reverse=True)

    selected = []
    total_tokens = 0
    for doc in ranked:
        if total_tokens >= tokens_target:
            break
        est_tokens = doc["char_len"] / chars_per_token
        doc["estimated_tokens"] = est_tokens
        selected.append(doc)
        total_tokens += est_tokens

    return selected, total_tokens
