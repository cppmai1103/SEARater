# Steps 7-8 — Weight Combinations and Weighted Selection

Code: [`scripts/generate_weight_combinations.py`](../scripts/generate_weight_combinations.py),
[`scripts/select_weighted_corpus.py`](../scripts/select_weighted_corpus.py)
Pipeline reference: `pipeline.md` Section 8 ("Generate 26 Candidate Weight
Combinations") and Section 7 ("Main method: Best weighted combination")

## Goal

Turn the scored candidate corpus (Step 5) into 26 concrete training
datasets — one per weight combination — that the proxy CPT step (Section 9)
will each continue-pretrain a fresh Qwen2.5-0.5B on.

Run in this order (also chained in `run_train_rater.sh`):

1. **`generate_weight_combinations.py`** (Section 8) — writes the fixed
   26-row weight table to `data/weight_combinations.json`.
2. **`select_weighted_corpus.py`** (Section 7) — for each of those 26
   weight vectors, ranks the scored corpus by weighted score and selects
   the top documents per language, up to a token budget.

## Part A — `generate_weight_combinations.py`

Section 8 is explicit that the pilot uses a **fixed, hand-picked** table of
26 weight vectors — no random sampling, no LightGBM. This script just
materializes that table (an "Edu only" vector, a "Reasoning only" vector,
several heavier-on-one-dimension combos, a few 2-dimension blends, etc.)
into JSON so downstream scripts can load it programmatically:

1. `WEIGHT_COMBINATIONS`: the 26 `(id, edu, reasoning, professionalism,
   cleanliness, cultural, meaning)` tuples, copied verbatim from
   pipeline.md's table.
2. `build_combinations()`: converts each tuple into `{"id", "meaning",
   "weights"}`, asserting the 5 weights sum to exactly 1.0 and none are
   negative — a sanity check on the table itself, not a computation.
3. `main()`: writes the list to `data/weight_combinations.json` and prints
   every combo for a quick visual check.

Verified by running it directly: all 26 combinations wrote correctly and
passed the sum-to-1.0 / non-negative assertions.

```bash
python3 scripts/generate_weight_combinations.py
```

## Part B — `select_weighted_corpus.py`

For each weight combination and language, computes:

```text
score = w1*educational_value + w2*reasoning + w3*professionalism
      + w4*cleanliness + w5*cultural_nuance
```

per document, ranks by that score, and keeps the top documents until a
token budget is hit — this *is* "Data selected by the weight combination"
that Section 9 trains on.

### 1. `join_scores_with_text(lang)`

The scored corpus (`data/candidate_corpus/scored/{lang}.jsonl`, from
`score_candidate_corpus.py`) only has `doc_id` + the 5 scores, not the
document text — kept lean since it can hold up to 100K rows. This function
joins each scored row back to its `text`/`char_len` **and** Section 5's
cheap prefilter features (`language_score`, `repetition_score`,
`target_script_ratio`, `stopword_ratio`, `symbol_ratio`, `numeric_ratio`,
`is_latin` — all computed in `build_candidate_corpus.py` via
`sea_rater/heuristics.py`, since none of them need a GPU or the rater)
from `data/candidate_corpus/{lang}.jsonl` by `doc_id`, and warns (rather
than failing) if a scored doc has no matching corpus row.

### 2. `select_for_weight(docs, weights, tokens_target, chars_per_token)`

1. Computes `weighted_score` for every document as the dot product of the
   weight vector and the 5 scores.
2. Sorts documents descending by `weighted_score`.
3. Walks the ranked list, accumulating an estimated token count
   (`char_len / chars_per_token`, default 4 chars/token) per document,
   stopping as soon as the running total reaches `tokens_target`.

Same cheap chars-per-token heuristic used elsewhere in the pipeline (e.g.
`build_validation_set.py`) — good enough to decide *how many* documents to
include, not meant as an exact token count.

### 3. `main()`

For every `(weight_id, language)` pair: takes a fresh copy of that
language's joined docs (since scoring mutates them in place and different
weight vectors would otherwise clobber each other's `weighted_score`),
runs the selection, and writes
`data/candidate_corpus/selected/{weight_id}/{lang}.jsonl`:

```json
{
  "doc_id": "<urn:uuid:...>",
  "language": "vi",
  "source": "clean",
  "weighted_score": 3.42,
  "estimated_tokens": 350.25,
  "text": "..."
}
```

Unlike the scored-corpus file, this one **does** include the full `text`
— it's what `run_proxy_cpt.py` tokenizes and trains on directly, so it
needs to be self-contained.

```bash
python3 scripts/select_weighted_corpus.py
python3 scripts/select_weighted_corpus.py --weight-ids W01 W06 --languages vi  # quick test
```

### What a real run needs

Both scripts are pure CPU/data-processing (no GPU, no model), but
`select_weighted_corpus.py` depends on `score_candidate_corpus.py` having
already produced `data/candidate_corpus/scored/{lang}.jsonl`. As of the
last check, that's real for vi/id/th/km (the original 4 languages) but
still missing for the 4 languages added when the pilot expanded to 8
(`ms`/`tl`/`my`/`lo`). Verified against the real vi data: `select_for_weight`
with W01/W06 correctly selects ~2M-token subsets. The join-and-rank logic
was also validated against a small synthetic fixture (20 fake documents,
one language) before being wired into the real pipeline: an edu-only
weight vector correctly selected exactly the top-N highest-
`educational_value` documents needed to hit a small test token budget.

**Known gotcha**: `data/candidate_corpus/{vi,id,th,km}.jsonl` on disk right
now predate `sea_rater/heuristics.py`'s cheap-feature fields
(`repetition_score`, `target_script_ratio`, ...), since they were built
before those were added. `select_for_weight` (used by `best_weighted`,
`edu_only`, `equal_average`) doesn't need those fields and works fine on
this old data, but `select_clean_heuristic` does — running it against
these 4 languages today silently selects **0 documents** (verified) rather
than erroring, since every doc fails the `repetition_score`/
`target_script_ratio` filters via their `.get()` fallback to `None`.
`build_final_cpt_dataset.py` now warns loudly when a baseline selects 0
docs for a language, specifically calling out this scenario, but the real
fix is rerunning `build_candidate_corpus.py` for those 4 languages to
backfill the fields.

## Re-running

```bash
python3 scripts/generate_weight_combinations.py
python3 scripts/select_weighted_corpus.py
```

Both are chained into `run_train_rater.sh` right after
`score_candidate_corpus.py`. Output lands in `data/weight_combinations.json`
and `data/candidate_corpus/selected/{weight_id}/{lang}.jsonl` (all under
`/data`, gitignored).
