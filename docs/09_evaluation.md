# Step 13 — Evaluation

Code: [`scripts/evaluate_models.py`](../scripts/evaluate_models.py)
Pipeline reference: `pipeline.md` Section 13 ("Evaluation")

## Goal

Answer the pilot's actual research question: does
`Qwen2.5-1.5B-CPT-BestWeighted` (Steps 11-12) beat the untouched
`Original Qwen2.5-1.5B Base`? Two evaluations, both comparing the same two
models:

1. **13.1 Held-out language modeling evaluation** — perplexity on the
   fixed validation set, fully implemented, torch-based.
2. **13.2 SEA downstream evaluation** — accuracy/F1 on real tasks (topic
   classification, MCQ) via `lm-evaluation-harness`, implemented as a
   best-effort CLI wrapper (see caveats below).

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
2. For each of the 4 languages, loads `data/validation_set/{lang}.jsonl`
   (the same fixed set used throughout the proxy search — comparability
   depends on this never changing), packs it with each model's tokenizer
   (identical tokenizer for both, since LoRA doesn't touch tokenization),
   and computes loss via the shared `evaluate_loss`.
3. Converts loss to perplexity (`exp(loss)`) per language per model.
4. Computes `macro_average_perplexity` (mean across 4 languages) and
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
`lm_eval` reports (accuracy, F1, etc., depending on the task).

```bash
python3 scripts/evaluate_models.py --device cuda   # runs both 13.1 and 13.2
```

### Caveats — this part is best-effort

Unlike everything else in this pipeline, 13.2 depends on an external
benchmark harness whose task registry this project doesn't control:

- **Task names may not match.** `sib200` and `belebele` are passed exactly
  as pipeline.md names them, but `lm-evaluation-harness` might register
  them under different task-group names (e.g. per-language variants like
  `sib200_vie_Latn`) depending on the installed version. Run
  `lm_eval --tasks list` on the cluster and adjust `--tasks` if the
  defaults don't resolve.
- **Per-language/macro breakdown isn't computed here.** Section 13's
  "Report" list under 13.2 asks for per-language, macro-average, and
  worst-language scores, matching 13.1's structure — but that requires
  knowing which per-task keys in `lm_eval`'s output correspond to which
  language, which depends on the exact task names resolved above. This
  script currently just passes through whatever `lm_eval` reports
  per-task; computing the per-language rollup is a small follow-up once
  the real task names are confirmed on the cluster.
- **Not runnable here at all** — no GPU, no `lm-evaluation-harness`
  installed on this dev server, and no trained models to evaluate yet.
  Syntax-checked only.

## Output

`data/evaluation_results.json`:

```json
{
  "held_out_lm_eval": {
    "base": {"vi": {"loss": ..., "perplexity": ...}, ..., "macro_average_perplexity": ..., "worst_language_perplexity": ...},
    "cpt_best_weighted": {"...": "..."},
    "relative_improvement_macro_perplexity": 0.05
  },
  "downstream_eval": {
    "base": {"sib200": {"...": "..."}, "belebele": {"...": "..."}},
    "cpt_best_weighted": {"...": "..."}
  }
}
```

## Re-running

```bash
python3 scripts/evaluate_models.py --device cuda
# or as part of the full final-CPT job:
sbatch scripts/run_final_cpt.sh
```
