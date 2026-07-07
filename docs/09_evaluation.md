# Step 13 — Evaluation

Code: [`scripts/evaluate_models.py`](../scripts/evaluate_models.py)
Pipeline reference: `pipeline.md` Section 13 ("Evaluation")

## Goal

Answer the pilot's actual research question: does
`Qwen2.5-1.5B-CPT-BestWeighted` (Steps 11-12) beat the untouched
`Original Qwen2.5-1.5B Base` — and optionally, how does it compare against
Section 7's other baselines (`random`, `clean_heuristic`, `equal_average`),
for whichever of those you've also trained via
`run_final_cpt.py --baseline ...`? `--baselines` (default `best_weighted`
only) picks which trained adapters to include; any baseline without a
trained adapter directory under `models/` is skipped with a warning rather
than failing the whole run. Two evaluations, both comparing the base model
against every requested baseline:

1. **13.1 Held-out language modeling evaluation** — perplexity on the
   fixed validation set, fully implemented, torch-based.
2. **13.2 SEA downstream evaluation** — accuracy/F1 on real tasks (topic
   classification, MCQ) via `lm-evaluation-harness`. `sib200` is fully
   wired up with a per-language/macro/worst-language accuracy rollup;
   `belebele` is a best-effort passthrough (see caveats below).

(13.3 "Forgetting/general ability evaluation" is commented out in
pipeline.md for this pilot, so there's no corresponding code.)

## Part A — 13.1: Held-out language modeling evaluation

### `load_lora_model(base_model_name, adapter_dir, device)`

Loads the base Qwen2.5-1.5B, then wraps it with `peft.PeftModel.from_pretrained(adapter_dir)`
to attach the trained LoRA adapter from `run_final_cpt.py` — this is how
the "CPT-BestWeighted" model actually gets reconstructed for evaluation
(the adapter alone isn't a runnable model; it has to be applied on top of
the same base checkpoint it was trained from).

### `run_held_out_eval(args)`

1. Loads both models: `model_base` (plain Qwen2.5-1.5B) and `model_cpt`
   (Qwen2.5-1.5B + LoRA adapter).
2. For each of the 8 languages, loads `data/validation_set/{lang}.jsonl`
   (the same fixed set used throughout the proxy search — comparability
   depends on this never changing), packs it with each model's tokenizer
   (identical tokenizer for both, since LoRA doesn't touch tokenization),
   and computes loss via the shared `evaluate_loss`.
3. Converts loss to perplexity (`exp(loss)`) per language per model.
4. Computes `macro_average_perplexity` (mean across 8 languages) and
   `worst_language_perplexity` (max) for each model.
5. Computes `relative_improvement_macro_perplexity =
   (base - cpt) / base` — positive means the CPT model is better.

Matches pipeline.md's requested report items (per-language perplexity,
macro-average, worst-language, relative improvement) except "Relative
improvement over Random CPT," since the Random baseline isn't part of this
pilot's active 2-model comparison (Section 11 dropped it along with the
other 3 baselines).

```bash
python3 scripts/evaluate_models.py --skip-downstream --device cuda  # 13.1 only
```

## Part B — 13.2: SEA downstream evaluation

### `run_lm_eval(model_args, output_path, tasks, device)`

Shells out to the `lm_eval` CLI (from `lm-evaluation-harness`, added to
`requirements.txt`):

```bash
lm_eval --model hf --model_args pretrained=... --tasks sib200,belebele \
        --device cuda --output_path ...
```

For the base model, `model_args` is just `pretrained=Qwen/Qwen2.5-1.5B`.
For the CPT model, it's `pretrained=Qwen/Qwen2.5-1.5B,peft=<adapter_dir>` —
`lm-evaluation-harness` has built-in support for loading a PEFT adapter on
top of a base model via that `model_args` string, so no custom model
loading code is needed here, just the right CLI invocation.

### `run_downstream_eval(args)`

Runs `run_lm_eval` once for each model against `--tasks` (default
`sib200 belebele`), parses each run's `results*.json`, and returns
`{"base": {...}, "cpt_best_weighted": {...}}` — whatever per-task metrics
`lm_eval` reports (accuracy, F1, etc., depending on the task), plus a
computed `sib200_summary` (see below) folded into each model's entry.

```bash
python3 scripts/evaluate_models.py --device cuda   # runs both 13.1 and 13.2
```

### `sib200`: wired up per-language, not just passed through

[Davlan/sib200](https://huggingface.co/datasets/Davlan/sib200) is a
204-language topic-classification benchmark; `lm-evaluation-harness`
registers it as one subtask per FLORES-200 language code (e.g.
`sib200_vie_Latn`), grouped under the plain `sib200` tag. Those FLORES-200
codes are exactly the same ones already used for `LANGUAGE_HF_CONFIGS`
(FineWeb2 uses the same taxonomy), so:

- **`expand_tasks(tasks)`**: replaces a bare `"sib200"` in `--tasks` with
  `SIB200_TASKS_BY_LANGUAGE.values()` — the 8 per-language subtask names
  — before calling `lm_eval`, so only our 8 languages run instead of all
  ~200 the full group tag would otherwise expand to. Any other requested
  task (`belebele`) passes through untouched.
- **`summarize_sib200(lm_eval_results)`**: pulls `acc,none` (falling back
  to `acc`) out of each `sib200_<flores_code>` entry in the raw per-task
  results, keyed back to our 2-letter language codes, and computes
  `macro_average_accuracy` / `worst_language_accuracy` — matching Section
  13.2's report spec (per-language, macro-average, worst-language score)
  the same way 13.1 already does for perplexity. Returns `None` if `sib200`
  wasn't requested, so it's a no-op when only `belebele` is run.
- The result is attached as `results[<model>]["sib200_summary"]` alongside
  the raw per-subtask entries (kept too, in case you need one language's
  exact numbers).

### Caveats — `belebele` is still best-effort

Unlike `sib200`, `belebele`'s exact per-language task-naming convention in
`lm-evaluation-harness` hasn't been verified against an installed copy, so
it's passed through as the plain `"belebele"` group tag without expansion
or a computed rollup:

- **Task name may not match.** Run `lm_eval --tasks list` on the cluster
  and adjust `--tasks` if `belebele` isn't registered under that exact
  name, or apply the same `expand_tasks`/`summarize_*` pattern used for
  `sib200` once its per-language subtask names are confirmed.
- **Not runnable here at all** — no GPU, no `lm-evaluation-harness`
  installed on this dev server, and no trained models to evaluate yet.
  Syntax-checked only; the `sib200` task-name assumption (FLORES-200 codes)
  is likewise unverified against a real installed harness.

## Output

`data/evaluation_results.json`:

```json
{
  "held_out_lm_eval": {
    "base": {"vi": {"loss": ..., "perplexity": ...}, ..., "macro_average_perplexity": ..., "worst_language_perplexity": ...},
    "best_weighted": {"...": "...", "relative_improvement_vs_base": 0.05},
    "random": {"...": "..."}
  },
  "downstream_eval": {
    "base": {
      "sib200_vie_Latn": {"acc,none": 0.62, "...": "..."},
      "sib200_ind_Latn": {"acc,none": 0.58, "...": "..."},
      "...": "... (one entry per sib200 language, plus belebele)",
      "sib200_summary": {
        "per_language_accuracy": {"vi": 0.62, "id": 0.58, "...": "..."},
        "macro_average_accuracy": 0.60,
        "worst_language_accuracy": 0.51
      }
    },
    "best_weighted": {"...": "...", "sib200_summary": {"...": "..."}}
  }
}
```

Each requested `--baselines` entry gets its own top-level key (instead of
one fixed `cpt_best_weighted` key) in both `held_out_lm_eval` and
`downstream_eval`; a baseline with no trained adapter simply doesn't appear.

## Re-running

```bash
python3 scripts/evaluate_models.py --baselines best_weighted random equal_average --device cuda
# or as part of the full final-CPT job:
sbatch scripts/run_final_cpt.sh
```
