"""
Shared selection logic, used by both the proxy selection
(scripts/select_weighted_corpus.py, Section 7) and the final CPT dataset
build (scripts/build_final_cpt_dataset.py, Section 11): weighted ranking
(one of 26 fixed combos, the single best_weight.json winner, or the fixed
equal-average weight vector) plus the two non-rater baselines (random,
clean-only heuristic) from Section 7.
"""

import json
import random
from pathlib import Path

from sea_rater.baselines import BASELINE_CHOICES
from sea_rater.dimensions import DIMENSIONS

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "data" / "candidate_corpus"
SCORED_DIR = CORPUS_DIR / "scored"

# pipeline.md Section 7 Baseline 3 is just a fixed weight vector, so it
# reuses select_for_weight below rather than needing its own function.
EQUAL_AVERAGE_WEIGHTS = {dim: 1.0 / len(DIMENSIONS) for dim in DIMENSIONS}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# Cheap prefilter fields (pipeline.md Section 5) computed in
# build_candidate_corpus.py, carried through from candidate_corpus/{lang}.jsonl
# rather than scored_corpus/{lang}.jsonl -- they need no GPU/rater, so they
# live on the corpus row, not the rater's score row.
CHEAP_FEATURE_FIELDS = [
    "language_score",
    "repetition_score",
    "target_script_ratio",
    "stopword_ratio",
    "symbol_ratio",
    "numeric_ratio",
    "is_latin",
]


def join_scores_with_text(lang):
    """Join scored_corpus/{lang}.jsonl (doc_id + 5 scores) back to its text
    and cheap prefilter features in candidate_corpus/{lang}.jsonl by doc_id."""
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
        joined.append(
            {
                **row,
                "text": corpus_row["text"],
                "char_len": corpus_row["char_len"],
                **{field: corpus_row.get(field) for field in CHEAP_FEATURE_FIELDS},
            }
        )
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


def select_random(docs, tokens_target, chars_per_token, seed=42):
    """pipeline.md Section 7 Baseline 1: sample documents in random order,
    independent of any rater score, until the token budget is met."""
    shuffled = docs.copy()
    random.Random(seed).shuffle(shuffled)

    selected = []
    total_tokens = 0
    for doc in shuffled:
        if total_tokens >= tokens_target:
            break
        est_tokens = doc["char_len"] / chars_per_token
        doc["estimated_tokens"] = est_tokens
        selected.append(doc)
        total_tokens += est_tokens

    return selected, total_tokens


def select_clean_heuristic(
    docs,
    tokens_target,
    chars_per_token,
    min_language_score=0.90,
    max_repetition_score=0.10,
    min_target_script_ratio=0.90,
):
    """pipeline.md Section 7 Baseline 2: cheap filters only, no rater score,
    no length window (the candidate corpus is only ever hard-cleaned by the
    bad-word filter in build_candidate_corpus.py -- this baseline's job is
    to test simple heuristic thresholds on top of that same corpus, not to
    re-clean it by length). Uses sea_rater/heuristics.py's computed fields:
    `language_score` (FineWeb2's langid confidence) for `langid_conf`, and
    `target_script_ratio` for `script_integrity`. Passing documents are kept
    in their original (unranked) order.
    """
    filtered = []
    for doc in docs:
        if doc.get("language_score") is None or doc["language_score"] < min_language_score:
            continue
        if doc.get("repetition_score") is None or doc["repetition_score"] > max_repetition_score:
            continue
        if doc.get("target_script_ratio") is None or doc["target_script_ratio"] < min_target_script_ratio:
            continue
        est_tokens = doc["char_len"] / chars_per_token
        doc["estimated_tokens"] = est_tokens
        filtered.append(doc)

    selected = []
    total_tokens = 0
    for doc in filtered:
        if total_tokens >= tokens_target:
            break
        selected.append(doc)
        total_tokens += doc["estimated_tokens"]

    return selected, total_tokens


def select_for_baseline(baseline, docs, tokens_target, chars_per_token, best_weights=None):
    """Dispatch to the right selection function for one of Section 7's
    baselines, or the main best_weighted method (`best_weights` required
    only in that case)."""
    if baseline == "random":
        return select_random(docs, tokens_target, chars_per_token)
    if baseline == "clean_heuristic":
        return select_clean_heuristic(docs, tokens_target, chars_per_token)
    if baseline == "equal_average":
        return select_for_weight(docs, EQUAL_AVERAGE_WEIGHTS, tokens_target, chars_per_token)
    if baseline == "best_weighted":
        return select_for_weight(docs, best_weights, tokens_target, chars_per_token)
    raise ValueError(f"unknown baseline {baseline!r}, expected one of {BASELINE_CHOICES}")
