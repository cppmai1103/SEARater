# Steps 11-12 — Final Dataset and Final Continued Pretraining

Code: [`scripts/build_final_cpt_dataset.py`](../scripts/build_final_cpt_dataset.py),
[`sea_rater/cpt.py`](../sea_rater/cpt.py) (LoRA additions),
[`scripts/run_final_cpt.py`](../scripts/run_final_cpt.py)
Pipeline reference: `pipeline.md` Section 11 ("Final Continued Pretraining
Datasets") and Section 12 ("Final Continued Pretraining")

## Goal

Take the single winning weight combination from Section 10
(`data/best_weight.json`) and actually build the real, full-scale model:
LoRA continue-pretrain Qwen2.5-1.5B Base into
`Qwen2.5-1.5B-CPT-BestWeighted` on 50M tokens (12.5M/language) selected by
that weight vector.

The pilot's other final "dataset" — the untouched original Qwen2.5-1.5B
Base — needs no data and no training; it's evaluated as-is in Step 13.

## Part A — `build_final_cpt_dataset.py` (Section 11)

Structurally almost identical to `select_weighted_corpus.py` (Section 7),
and deliberately so: this step reuses the exact same ranking algorithm,
just with two differences reflected in the refactor described below:

- **One weight vector, not 16.** Loads `data/best_weight.json` (Section
  10's output) instead of iterating `data/weight_combinations.json`.
- **A much bigger token budget.** 12.5M tokens/language (50M total) instead
  of the proxy's 2M/language — this is training the real final model, not
  a cheap proxy.

### The shared refactor: `sea_rater/selection.py`

`join_scores_with_text` (join scored docs back to their text by `doc_id`)
and `select_for_weight` (rank by weighted score, take top docs until a
token budget is hit) used to live inside `select_weighted_corpus.py`.
Since this script needed the identical logic, they were pulled out into
`sea_rater/selection.py` so both scripts import the same code rather than
duplicating it. `select_weighted_corpus.py` was updated to import from
there too — re-verified afterward with the same synthetic-fixture test
used originally (10 fake documents, edu-only weight, correct top-2
selection for a small token target) to confirm the move didn't change
behavior.

### `main()`

For each language: join scores with text, rank and select by the best
weight up to `--tokens-per-language` (default 12.5M), write
`data/candidate_corpus/final_cpt_dataset/{lang}.jsonl` — same record
schema as the proxy selection (`doc_id`, `language`, `source`,
`weighted_score`, `estimated_tokens`, `text`).

```bash
python3 scripts/build_final_cpt_dataset.py
```

## Part B — LoRA support in `sea_rater/cpt.py`

Two additions to the shared CPT utilities module used by both the proxy
run (Section 9) and this final run:

- `FINAL_MODEL_NAME = "Qwen/Qwen2.5-1.5B"` alongside the existing
  `PROXY_MODEL_NAME = "Qwen/Qwen2.5-0.5B"`.
- `apply_lora(model, r, lora_alpha, lora_dropout, target_modules)`: wraps
  a loaded model with `peft.LoraConfig` + `get_peft_model`, targeting
  Qwen2's standard attention/MLP projections (`q_proj`, `k_proj`, `v_proj`,
  `o_proj`, `gate_proj`, `up_proj`, `down_proj`). This freezes the base
  model and adds small trainable adapter matrices — only those end up with
  `requires_grad=True`.
- `trainable_parameters(model)`: a small helper so the optimizer only ever
  sees the LoRA adapter weights, not the frozen 1.5B base.

## Part C — `run_final_cpt.py` (Section 12)

1. Loads a fresh Qwen2.5-1.5B Base + tokenizer, wraps it with
   `apply_lora` (default `r=16`, `lora_alpha=32`, `lora_dropout=0.05`),
   and prints the trainable-parameter count as a sanity check that LoRA is
   actually only exposing a small adapter, not the full 1.5B model.
2. Loads all 4 languages' `data/candidate_corpus/final_cpt_dataset/{lang}.jsonl`
   (Part A's output) and combines them into one text list.
3. Packs into `--seq-length`-token blocks (same `pack_texts` used by the
   proxy run) and trains for `--epochs` (default 1) with `AdamW` over only
   `trainable_parameters(model)` — the frozen base never gets gradient
   updates.
4. Saves the LoRA adapter (not the full model — LoRA saves are small,
   typically tens of MB rather than the base model's several GB) plus the
   tokenizer to `models/qwen1.5b_cpt_bestweighted/`.

```bash
python3 scripts/run_final_cpt.py --device cuda
```

## What's actually been tested

Nothing in this file could be run end-to-end here — no GPU, no
`torch`/`transformers`/`peft` on this dev server, and no real
`data/best_weight.json` yet (the proxy CPT run that produces it hasn't
been executed on the cluster). `build_final_cpt_dataset.py`'s *selection
logic* is the same code already validated via `select_weighted_corpus.py`'s
synthetic-fixture test; `run_final_cpt.py`'s LoRA/training code is
syntax-checked only (`py_compile`), not run.

## Re-running

```bash
python3 scripts/build_final_cpt_dataset.py
python3 scripts/run_final_cpt.py --device cuda
# or as one cluster job:
sbatch scripts/run_final_cpt.sh
```

`run_final_cpt.sh` assumes `data/best_weight.json` already exists (Step
9-10's `run_proxy_cpt.sh` must have completed first). Output lands in
`data/candidate_corpus/final_cpt_dataset/{lang}.jsonl` and
`models/qwen1.5b_cpt_bestweighted/` (both gitignored).
