# Steps 9-10 — Proxy Continued Pretraining and Best Weight Selection

Code: [`scripts/build_validation_set.py`](../scripts/build_validation_set.py),
[`sea_rater/cpt.py`](../sea_rater/cpt.py),
[`scripts/run_proxy_cpt.py`](../scripts/run_proxy_cpt.py),
[`scripts/select_best_weight.py`](../scripts/select_best_weight.py)
Pipeline reference: `pipeline.md` Section 9 ("Proxy Continued Pretraining")
and Section 10 ("Select Final Weight Combination")

## Goal

Answer the question the whole pipeline has been building toward: of the 16
weight combinations (Step 7-8), which one actually produces the best
continued-pretraining data? Answered empirically, not by inspection —
cheaply continue-pretrain a small proxy model on each combination's
selected data, evaluate all 16 on the same held-out validation set, and
pick the winner by validation loss.

Run in this order (also chained in `run_proxy_cpt.sh`):

1. **`build_validation_set.py`** — build the one fixed validation set every
   proxy run (and later the final run) will be judged against.
2. **`run_proxy_cpt.py`** (Section 9) — 16 independent continue-pretraining
   runs, one per weight combination, each evaluated on that validation set.
3. **`select_best_weight.py`** (Section 10) — pick the winner from the 16
   results.

## Part A — `build_validation_set.py`

A held-out set has to exist *before* any proxy training happens, and
Section 9 is explicit it "must not overlap with candidate training data."

The candidate corpus (`build_candidate_corpus.py`) streams the *first*
100K documents per language (80K clean / 20K removed) from each FineWeb2
split. This script streams the same splits again but **skips** exactly
that many documents first (`SKIP_CLEAN = 80_000`, `SKIP_REMOVED = 20_000`,
computed from the same `CANDIDATE_POOL_SIZE`/`CANDIDATE_CLEAN_RATIO`
constants candidate-corpus building uses), then collects new documents
until hitting a token target — 1M tokens/language, 4M total, 80/20
clean/removed, estimated the same `char_len / chars_per_token` way as
`select_weighted_corpus.py`. Guarantees no overlap without ever needing to
diff two multi-hundred-thousand-row ID sets against each other.

```bash
python3 scripts/build_validation_set.py
```

Output: `data/validation_set/{lang}.jsonl`.

## Part B — `sea_rater/cpt.py`: shared training/eval utilities

Used by both the proxy run (Section 9) and, later, the final 1.5B CPT run
(Section 12), so it lives in the shared package rather than a script:

- `load_cpt_model_and_tokenizer(model_name, device)`: loads Qwen2.5-0.5B
  (bfloat16) and its tokenizer.
- `pack_texts(texts, tokenizer, seq_length)`: tokenizes every document,
  appends an EOS token after each one, concatenates everything into one
  long id stream, then chunks it into fixed-`seq_length` blocks (dropping
  the final partial block) — the standard packed-pretraining layout, so
  training doesn't waste compute on padding.
- `train_one_epoch` / `evaluate_loss`: a plain PyTorch loop over packed
  blocks (`labels=input_ids`, causal LM loss from `transformers`), the
  latter under `torch.no_grad()`.

## Part C — `run_proxy_cpt.py` (Section 9)

For each of the 16 weight combinations, independently:

1. **Fresh checkpoint.** Loads a brand-new `Qwen2.5-0.5B` copy — critical
   for a fair comparison; no run ever continues from a previous run's
   trained weights.
2. **Load that combo's data.** Reads all 4 languages'
   `data/candidate_corpus/selected/{weight_id}/{lang}.jsonl` (Step 7's
   output) and concatenates them — language balance is already satisfied
   per-file, since each language was independently token-budgeted there.
3. **Pack and train one epoch.** Because every weight combination's
   selected data was sized to the same ~8M-token budget, packing +
   training "one epoch" naturally uses the same token budget and step
   count for every run, without needing to separately hardcode a step
   count (pipeline.md's "keep training steps fixed" falls out of this by
   construction rather than needing to be enforced explicitly).
4. **Evaluate on the fixed validation set.** The validation blocks are
   packed once (memoized in `validation_loaders_factory`, since every run
   uses the identical Qwen2.5-0.5B tokenizer) and reused, unchanged, across
   all 16 evaluations — so differences in loss reflect the training data,
   not the evaluation data.
5. **Record the row**: `weight_id`, the 5 weights, `loss_vi`/`loss_id`/
   `loss_th`/`loss_km`, `macro_loss` (mean of the 4), `worst_language_loss`
   (max of the 4) — appended to `data/proxy_results.csv` immediately after
   each run finishes (not batched at the end), so a crash partway through
   doesn't lose completed runs.
6. **Free GPU memory** (`del model`, `torch.cuda.empty_cache()`) before the
   next weight combination.

```bash
python3 scripts/run_proxy_cpt.py --device cuda
python3 scripts/run_proxy_cpt.py --weight-ids W01 W06 --device cuda  # quick test, 2 combos only
```

## Part D — `select_best_weight.py` (Section 10)

Pure CSV/JSON logic, no model or GPU involved:

1. Load `proxy_results.csv`, find the minimum `macro_loss`.
2. Collect every combo within `--tie-tolerance` (default 1%, relative) of
   that minimum as "contenders" — pipeline.md's "if two weights are very
   close" turned into a concrete rule.
3. Among the contenders, pick the one with the lowest `worst_language_loss`
   — this is a no-op if there's only one contender (the clear winner).
4. Look up that `weight_id`'s full weight vector and `meaning` from
   `data/weight_combinations.json`, and write everything to
   `data/best_weight.json`: the id, meaning, weights, and all of its
   metrics, for provenance.

```bash
python3 scripts/select_best_weight.py
```

### What's actually been tested

`select_best_weight.py` was fully validated locally with a synthetic
16-row `proxy_results.csv` (random per-language losses, one combo forced
into a 3-way tie for lowest `macro_loss`): it correctly identified the tie
group and broke it by `worst_language_loss`.

`build_validation_set.py`, `sea_rater/cpt.py`, and `run_proxy_cpt.py`
could **not** be run end-to-end here — no GPU, no `torch`/`transformers`
installed on this dev server, and no real Qwen2.5-0.5B/selected-data/
validation-set artifacts exist yet (Step 7's `select_weighted_corpus.py`
run is currently commented out in `run_train_rater.sh`, so
`data/candidate_corpus/selected/` doesn't exist yet either). These were
written and `py_compile`-checked for syntax only; the actual training loop
needs a real cluster run before trusting its numbers.

## Re-running

```bash
python3 scripts/build_validation_set.py
python3 scripts/run_proxy_cpt.py --device cuda
python3 scripts/select_best_weight.py
# or as one cluster job:
sbatch scripts/run_proxy_cpt.sh
```

`run_proxy_cpt.sh` assumes Steps 7-8 (`generate_weight_combinations.py`,
`select_weighted_corpus.py`) have already produced
`data/candidate_corpus/selected/` — it doesn't run them itself.

Output lands in `data/validation_set/{lang}.jsonl`, `data/proxy_results.csv`,
and `data/best_weight.json` (all under `/data`, gitignored).
