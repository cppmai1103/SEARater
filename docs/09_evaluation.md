# Step 13 — Evaluation

Code: [`scripts/build_downstream_eval_data.py`](../scripts/build_downstream_eval_data.py),
[`scripts/evaluate_models.py`](../scripts/evaluate_models.py)
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
than failing the whole run.

1. **13.1 Held-out language modeling evaluation** — perplexity on the
   fixed validation set.
2. **13.2 SEA downstream evaluation** — 0-shot (or k-shot) multiple-choice
   accuracy on `Davlan/sib200` (topic classification, per-language),
   `facebook/belebele` (reading comprehension, per-language), and
   `cais/mmlu` (general-ability check, per-subject, English-only — this is
   13.3 "Forgetting/general ability evaluation" folded in here rather than
   left as a separate unimplemented section).

## Two scripts, not one — data prep vs. scoring

`build_downstream_eval_data.py` loads sib200/belebele/mmlu from the
Hugging Face Hub, builds every doc's MCQ prompt (with any k-shot prefix
already baked in), and writes them to local JSONL. `evaluate_models.py`
then does *zero* dataset loading — it just reads that local JSONL and
scores it against each model. Splitting these apart means:

- Re-running eval (e.g. one baseline at a time, in a fresh process, for
  GPU-memory isolation — see "Known OOM failure modes" below) never
  re-downloads or re-tokenizes sib200/belebele/mmlu; each run just reads
  local files.
- `evaluate_models.py` needs no Hugging Face Hub access and no `datasets`
  library import at all — only `build_downstream_eval_data.py` does.
- The exact evaluated set (which docs, which k-shot exemplars) is pinned to
  disk and inspectable/diffable, rather than implicitly whatever the live
  HF dataset returns at the moment a job happens to run.

```bash
python3 scripts/build_downstream_eval_data.py                                    # once, 0-shot
python3 scripts/build_downstream_eval_data.py --num-fewshot 5 --mmlu-samples-per-subject 10
python3 scripts/evaluate_models.py --device cuda                                 # reads the above
```

## Why no `lm-evaluation-harness`

An earlier version of this script shelled out to the `lm_eval` CLI for
13.2. That was dropped after a real cluster run failed with `Tasks not
found: sib200_vie_Latn, ...` — checked directly against the installed
package (`lm-eval==0.4.12`): there is no `lm_eval/tasks/sib200/` directory
at all, so `sib200` was never a registered task in this harness version,
not a naming mismatch. `belebele` *is* registered (it wasn't in that run's
"not found" list) but shelling out to a separate CLI process per
model/baseline was slower and harder to debug than scoring in-process,
especially since 13.1 already loads each model once anyway — so all tasks
are now scored natively instead.

## Part A — `build_downstream_eval_data.py`

### The datasets, confirmed against their real schemas

`Davlan/sib200`, `facebook/belebele`, and `cais/mmlu` were all checked
directly against HF's `datasets-server` API rather than assumed — an
earlier assumption about `sib200`'s schema, and a separate assumption that
FineWeb2's `fil_Latn` language code would work for these benchmarks too,
both turned out to need correcting once actually checked:

- **`Davlan/sib200`**: `text` (str), `category` (str, one of the 7
  `SIB200_CATEGORIES`), `index_id` (int). `test` split (204 rows/language,
  used for eval), `train` split (used for `--num-fewshot` exemplars).
- **`facebook/belebele`**: `flores_passage` (str), `question` (str),
  `mc_answer1..4` (str), `correct_answer_num` (1-indexed string). `test`
  split only (no train/dev) — see the k-shot section below for how that
  affects `--num-fewshot`. Also has no Filipino/Tagalog config at all.
- **`cais/mmlu`** (config `"all"`): `question` (str), `subject` (str),
  `choices` (list[str], always 4), `answer` (int index into `choices`).
  `test` split (14,042 questions across 57 subjects, used for eval), `dev`
  split (5 rows/subject, the standard MMLU few-shot split, used for
  `--num-fewshot` exemplars).

sib200/belebele have one config per FLORES-200 language code — mostly the
same codes already used for `LANGUAGE_HF_CONFIGS` (FineWeb2 uses the same
taxonomy), with one exception: FLORES-200 has no separate Filipino code, so
FineWeb2's `fil_Latn` isn't valid for either benchmark (confirmed against a
real `BuilderConfig 'fil_Latn' not found` error) — `DOWNSTREAM_HF_CONFIGS`
substitutes `tgl_Latn` (Tagalog) for `tl` just for this section, leaving
`LANGUAGE_HF_CONFIGS` itself untouched since `fil_Latn` is correct for
corpus building.

`available_language_configs(dataset_path, hf_configs)` calls
`datasets.get_dataset_config_names(dataset_path)` once per dataset and
checks every one of our languages against the real result *before* loading
anything, so a language missing from one benchmark (belebele has no
Filipino/Tagalog config at all) is reported once (with the real available
list) and just skipped for that dataset, instead of crashing the whole run
on whichever missing config it happens to hit first.

### MCQ formatting and the length-bias problem it solves

`format_mcq_context(passage_and_question, choice_texts)` builds a real
lettered multiple-choice prompt:

```
<passage/question>
A. <choice>
B. <choice>
...
Select the one correct letter for the answer.
Answer:
```

Only the single letters (`MCQ_LETTERS`) get scored as continuations —
never each choice's raw text. Plain continuation-scoring (score the raw
answer text directly, no letters shown) has a real length bias: a short
wrong answer's total log-likelihood can beat a longer correct one purely
because it has fewer tokens to "clear." Concretely, for a 3-token correct
answer at ~50% per-token confidence vs. a 1-token wrong answer at ~35%
confidence, the wrong answer's raw summed log-likelihood (`log 0.35 ≈
-1.05`) can beat the correct one's (`log 0.5 × 3 ≈ -2.08`) even though the
model is *more* confident per-token in the right answer. Lettered choices
remove this entirely since every candidate becomes exactly one token — at
the cost of requiring the model to follow the lettered-option convention,
which is why the explicit "Select the one correct letter..." instruction
is there.

### k-shot exemplars (`--num-fewshot`)

`build_fewshot_prefix(exemplar_docs)` turns k exemplar docs into one
prefix string — each exemplar's full MCQ prompt with the correct letter
filled in after `Answer:`, separated by a blank line — prepended to the
real question. Sourced per task so there's never any leakage between
exemplars and evaluated docs:

- **sib200**: exemplars come from its real `train` split.
- **mmlu**: exemplars come from its dedicated `dev` split (5 rows/subject
  by design, the standard MMLU few-shot setup); clamped to whatever's
  actually available if more is requested than exists for a subject.
- **belebele**: has no separate train/dev split at all, so the first
  `--num-fewshot` rows of each language's own `test` set are reserved as
  exemplars and *excluded* from evaluation — evaluated doc count per
  language shrinks by `--num-fewshot` when it's `> 0`.

### Output layout

```
data/downstream_eval_data/sib200/{lang}.jsonl     -- one file per language
data/downstream_eval_data/belebele/{lang}.jsonl   -- one file per language
data/downstream_eval_data/mmlu.jsonl              -- one combined file,
                                                      every row tagged
                                                      with a "subject" field
```

Every row is `{"context": str, "choices": [str, ...], "choice_texts": [str, ...], "gold": int}`
(plus `"subject"` for mmlu rows) — exactly the shape
`evaluate_models.py`'s `evaluate_multiple_choice` expects, with zero
further processing needed.

```bash
python3 scripts/build_downstream_eval_data.py
python3 scripts/build_downstream_eval_data.py --num-fewshot 5 --mmlu-samples-per-subject 10
python3 scripts/build_downstream_eval_data.py --mmlu-limit 2000   # flat cap instead (can skip later subjects)
```

## Part B — `evaluate_models.py`: model loading

`load_lora_model(base_model_name, adapter_dir, device)` loads the base
Qwen2.5-1.5B, then wraps it with `peft.PeftModel.from_pretrained(adapter_dir)`
to attach the trained LoRA adapter from `run_final_cpt.py` — this is how
each "CPT-<baseline>" model actually gets reconstructed for evaluation (the
adapter alone isn't a runnable model; it has to be applied on top of the
same base checkpoint it was trained from). The untouched base model uses
the same `load_cpt_model_and_tokenizer` helper 13.1/Section 9/12 already
use. `evaluate_one_model` calls `model.eval()` explicitly before anything
else runs — LoRA adapters here use `lora_dropout=0.05` (nonzero), so
skipping this would make 13.2's accuracy/`token_normalized_prob_correct`
non-deterministic per run.

Each model (base + every requested baseline) is loaded exactly once, run
through both evaluations, then freed before the next model loads — and
`data/evaluation_results.json` is rewritten after every model finishes, not
just once at the end, so a crash partway through a multi-baseline run never
discards an already-evaluated model's numbers.

## Part C — 13.1: held-out language modeling evaluation

For each of the 8 languages, loads `data/validation_set/{lang}.jsonl` (the
same fixed set used throughout the proxy search — comparability depends on
this never changing), packs it with the model's tokenizer, and computes
loss via the shared `evaluate_loss` (wrapped in `torch.no_grad()`).
Converts loss to perplexity (`exp(loss)`) per language, then computes
`macro_average_perplexity` (mean across 8 languages) and
`worst_language_perplexity` (max). For every baseline (not the base model
itself), `add_relative_improvement` then adds `relative_improvement_vs_base
= (base - cpt) / base` — positive means the CPT model is better.

```bash
python3 scripts/evaluate_models.py --task perplexity --device cuda  # 13.1 only, no local eval data needed
```

## Part D — 13.2: SEA downstream evaluation

### `score_continuations` — the actual scorer

For a batch of parallel `(context, continuation)` pairs, tokenizes
`context + continuation` together, runs one forward pass, and sums the
model's log-probability of exactly the continuation's tokens (the tokens
after the context) — the same "loglikelihood" scoring
`lm-evaluation-harness` uses internally for `output_type: multiple_choice`
tasks. Sequences are right-padded to the batch's longest length; per-row
continuation start/length is tracked individually since rows in a batch
can have different context and continuation lengths. Returns
`(sum_log_prob, num_tokens)` per pair, not just the sum, so callers can
length-normalize.

`evaluate_multiple_choice(model, tokenizer, docs, ...)` scores every
`(doc["context"], " " + letter)` pair for every letter in every doc in one
batched call. Two metrics come out of that per doc:

- **`accuracy`**: the letter with the highest raw summed log-probability
  wins (argmax), compared against `doc["gold"]` — matches lm-eval's plain
  `"acc"`. Binary per doc (0 or 1).
- **`token_normalized_prob_correct`**: `exp(sum_log_prob_of_gold_choice /
  num_tokens_in_gold_choice)` — the model's average per-token probability
  for *specifically the correct answer*, regardless of whether it won the
  argmax. Unlike accuracy, this is continuous and still moves when a
  model's belief in the correct answer shifts without its top-1 pick
  flipping — useful for seeing a CPT model "getting warmer" on a task where
  raw accuracy looks flat between two checkpoints.

### Bucketing: language vs. subject

`BUCKET_LABEL_BY_TASK = {"sib200": "language", "belebele": "language", "mmlu": "subject"}`
— `rollup_downstream_task` is bucket-agnostic and used identically for all
three tasks; only the JSON key names (`per_language_accuracy` vs.
`per_subject_accuracy`) change based on this lookup. Both metrics are
rolled up to macro-average/worst-bucket, and `add_relative_improvement`
computes `relative_improvement_vs_base` for both.

### Per-doc predictions

`COMBINE_PREDICTIONS_BY_TASK = {"sib200": False, "belebele": False, "mmlu": True}`
— sib200/belebele save one predictions file per language (8 files, each a
manageable size); mmlu combines all 57 subjects into one file instead
(each row tagged with a `"subject"` field), since per-subject files would
mean 57 mostly-tiny files for a task that isn't part of the per-language
SEA rollup anyway.

```json
{"context": "...", "choices": ["science/technology", "..."], "gold": 0, "gold_letter": "A", "gold_choice": "science/technology", "predicted": 2, "predicted_letter": "C", "predicted_choice": "politics", "correct": false, "choice_log_likelihoods": [-9.1, -14.2, -7.8, "..."], "token_normalized_prob_correct": 0.0031}
```

This is what accuracy/probability numbers alone can't show: *which* docs a
model got wrong, what it guessed instead, and how close its distribution
over choices was — needed for any real error analysis, not just the
aggregate scores.

### Known OOM failure modes (and their fixes)

Four distinct memory failures have shown up in real cluster runs of this
pipeline, each with a different root cause:

1. **A single batch's forward pass is too big for the GPU** — the `logits`
   tensor's fp32 upcast (`[batch_size, seq_len, vocab_size]`) can be
   several GiB on its own for a large `--eval-batch-size`/
   `--downstream-batch-size` on a small-VRAM card. Fix: lower
   `--eval-batch-size`/`--downstream-batch-size` (batch size 1 is always
   safe — at `seq_len=1024`, the fp32 logits tensor is only ~0.6 GiB).
2. **Memory creeps up across many buckets within one model's evaluation**
   — confirmed via a real crash 6 languages into `belebele`, with far more
   "allocated by PyTorch" memory in use than a single forward pass should
   ever need. Same root cause already documented and fixed once in
   `run_proxy_cpt.py`: CUDA tensors can form reference cycles plain
   refcounting won't clear, so `gc.collect()` has to run before
   `torch.cuda.empty_cache()` has anything to release. Fixed:
   `rollup_downstream_task` now calls both after every bucket.
3. **Allocator fragmentation *within* one bucket's batch loop** — a
   different crash, mid-bucket (batch 90/179 of `sib200[my]`), with a
   large and *growing* "reserved by PyTorch but unallocated" figure across
   crashes (2.28 GiB → 4.27 GiB) — the OOM message's own fragmentation
   signal. Root cause: `score_continuations` batched `(context,
   continuation)` pairs in their original order, so each batch's padding
   target (and thus its logits tensor's size) swung unpredictably
   batch-to-batch depending on which docs happened to land together,
   fragmenting the caching allocator over many batches until an allocation
   that should've fit couldn't find a contiguous block. Fixed: pairs are
   now sorted by token length before batching (like lm-evaluation-harness
   does internally) and unsorted back to the caller's order before
   returning — allocation sizes change smoothly across the run instead of
   jumping randomly, which also cuts padding waste as a side effect.
4. **LoRA dropout active during eval** — not an OOM, but a related
   correctness gap found while investigating: `evaluate_loss` calls
   `model.eval()` internally, but 13.2's scoring path never did, and it
   only "worked" because 13.1 always ran first in the same process,
   leaving the model in eval mode as a side effect. Fixed: `model.eval()`
   is now called explicitly at the top of `evaluate_one_model`.
5. **One bucket's OOM used to kill the entire process** — confirmed by two
   real crashes (`job-2887`, `job-2897`) that both took down the whole
   `evaluate_models.py` run on belebele's `km`, discarding every language
   still queued after it (`lo`, `ms`, `my`, `th`, `tl`, `vi`) even though
   most of those fit fine on their own — belebele's passages vary a lot in
   length per language, so one language OOMing doesn't mean the next one
   will. Fixed: `evaluate_bucket_with_retry` catches
   `torch.OutOfMemoryError` per bucket, clears the cache, and retries that
   bucket at half the batch size (down to a floor of 1) before giving up on
   just that bucket and moving on — `rollup_downstream_task` then skips it
   (logged, and still absent from `--output-path`, so a later run's
   auto-skip retries only that bucket) instead of the whole model/task
   dying.

```bash
python3 scripts/evaluate_models.py --eval-batch-size 4 --downstream-batch-size 4 --device cuda
python3 scripts/evaluate_models.py --eval-batch-size 1 --downstream-batch-size 1 --device cuda  # safest floor
```

### `--task`: one task per process, for maximum isolation

Every crash above happened *inside* one task's own batch loop (perplexity,
or one of sib200/belebele/mmlu) — never caused by leftover state from a
*different* task. `--task {all,perplexity,sib200,belebele,mmlu}` (default
`all`) scopes one invocation to exactly one of them, so running each task
as its own fresh process gives the strongest possible isolation: a crash
(or any subtle memory buildup, known or not) in one task can never inherit
pressure from, or leak into, another, since the OS fully reclaims a
process's GPU memory when it exits.

This only works because `--output-path` is now **merged, not overwritten**:
`load_existing_results`/`merge_model_result` read whatever's already on
disk and merge this run's task result into the right model's entry, and
`add_relative_improvement` recomputes `relative_improvement_vs_base`
opportunistically for whatever tasks `"base"` and a baseline currently both
have — which may have landed in the file from two entirely different
processes, in either order. `load_downstream_docs` also only reads the
local JSONL for whichever task(s) this run actually needs.

```bash
# one process per task, all baselines together within each (still isolated by task)
for task in perplexity sib200 belebele mmlu; do
  python3 scripts/evaluate_models.py --baselines best_weighted random clean_heuristic equal_average \
      --task "$task" --device cuda --eval-batch-size 8 --downstream-batch-size 8
done
```

### Automatic OOM retry/skip + auto-skip: one command handles the rest of belebele

Per-bucket OOM retry (failure mode #5 above) and the default **auto-skip**
behavior together mean a single `--task belebele` invocation is enough —
no need to hand-pick which language to run first anymore:

- `evaluate_bucket_with_retry` catches a bucket's OOM, halves the batch
  size, and retries (down to a floor of 1) before giving up on just that
  one bucket, so one bad language can no longer take the whole run down
  with it.
- `filter_unevaluated_buckets` drops any (model, task, language) combo
  already present in `--output-path` before evaluating, per model tag (a
  baseline can have a different done-set than `base`) — so re-running the
  same command costs nothing for buckets already finished.
- `merge_downstream_task_result` (used by `merge_model_result`) merges
  per-language rather than replacing the whole task, so partial results
  from many separate invocations (or bucket skips within one invocation)
  accumulate correctly instead of clobbering each other.

`--languages <code> [<code> ...]` is still there if you want to scope a run
to specific codes manually (ignored for mmlu, bucketed by subject, not
language); `--force-recompute` disables auto-skip and redoes every
requested bucket regardless (e.g. after retraining an adapter, when the
old numbers for it are stale).

```bash
# one command: retries/skips whatever OOMs per-language, auto-skips whatever's already done
python3 scripts/evaluate_models.py --task belebele --eval-batch-size 8 --downstream-batch-size 8 \
    --baselines best_weighted random clean_heuristic equal_average --device cuda

# if anything still got skipped (OOM even at batch_size=1), just run the exact same
# command again -- auto-skip means it only retries what's still missing
python3 scripts/evaluate_models.py --task belebele --eval-batch-size 8 --downstream-batch-size 8 \
    --baselines best_weighted random clean_heuristic equal_average --device cuda

python3 scripts/evaluate_models.py --task mmlu --device cuda   # just re-run mmlu, e.g. after a crash
```

## Output

`data/evaluation_results.json`, keyed by model (not by section):

```json
{
  "base": {
    "held_out_lm_eval": {
      "vi": {"loss": 2.28, "perplexity": 9.75}, "...": "...",
      "macro_average_perplexity": 10.02, "worst_language_perplexity": 24.88
    },
    "downstream_eval": {
      "sib200": {
        "per_language_accuracy": {"vi": 0.62, "id": 0.58, "...": "..."},
        "macro_average_accuracy": 0.60, "worst_language_accuracy": 0.51,
        "per_language_token_normalized_prob_correct": {"vi": 0.31, "id": 0.28, "...": "..."},
        "macro_average_token_normalized_prob_correct": 0.29,
        "worst_language_token_normalized_prob_correct": 0.19
      },
      "belebele": {"per_language_accuracy": "...", "macro_average_accuracy": "...", "worst_language_accuracy": "..."},
      "mmlu": {"per_subject_accuracy": "...", "macro_average_accuracy": "...", "worst_subject_accuracy": "..."}
    }
  },
  "best_weighted": {
    "held_out_lm_eval": {"...": "...", "relative_improvement_vs_base": 0.204},
    "downstream_eval": {
      "sib200": {
        "...": "...",
        "relative_improvement_vs_base": 0.03,
        "token_normalized_prob_correct_relative_improvement_vs_base": 0.05
      },
      "belebele": {"...": "..."},
      "mmlu": {"...": "..."}
    }
  }
}
```

Each requested `--baselines` entry gets its own top-level key; a baseline
with no trained adapter simply doesn't appear. `relative_improvement_vs_base`
is only computed for baselines (never for `"base"` itself).

## Re-running

```bash
python3 scripts/build_downstream_eval_data.py --num-fewshot 5 --mmlu-samples-per-subject 10  # once
python3 scripts/evaluate_models.py --baselines best_weighted random equal_average --device cuda
```

`data/evaluation_results.json` is gitignored (under `/data`) and **merged**
into on every write (`load_existing_results`/`merge_model_result`), not
overwritten — re-running with a different `--baselines` list, or a
different `--task`, or from an entirely different process, only ever
updates the specific model/task keys this run actually computed, leaving
everything else already on disk untouched. Within a downstream task, the
merge is per-bucket (`merge_downstream_task_result`), and by default each
model (`"base"` included) skips any bucket it already has a result for
(`filter_unevaluated_buckets`) rather than always re-scoring everything —
pass `--force-recompute` when you actually want fresh numbers for buckets
that are already on disk (e.g. after retraining an adapter). See
"Automatic OOM retry/skip + auto-skip" above for the belebele-resume
workflow this enables.

`data/downstream_predictions/` is a per-model overwrite too:
`save_predictions`/`save_predictions_combined` truncate on every write, so
re-running the same `--baselines` list replaces those prediction files
rather than appending to them.

`data/downstream_eval_data/` only needs rebuilding when you want a
different `--num-fewshot`/`--mmlu-limit`/`--mmlu-samples-per-subject` — it
doesn't depend on which model is being evaluated, so it's safe to leave in
place across many `evaluate_models.py` runs.

## Results (as of the 2026-07-10 run)

`data/evaluation_results.json` currently has **perplexity, sib200, and
mmlu** complete for `base` and all four baselines (`best_weighted`,
`random`, `clean_heuristic`, `equal_average`). **belebele is missing** —
two attempts (`job-2887`, 2026-07-09, and `job-2897`, 2026-07-10) both
OOM'd on the `base` model right at the start of the `km` language pass
(batch 1/895 in the latest attempt, immediately after `id` finished
cleanly) — the same signature as "Known OOM failure modes" #1/#3 above,
just not yet hit with a lower `--eval-batch-size`/`--downstream-batch-size`
(both runs used the default of 16) — and, at the time, one bucket's OOM
still crashed the entire process (failure mode #5 above), so the whole
run was lost rather than just `km`. Now fixed: a plain `--task belebele`
re-run retries/skips whatever OOMs per-language and auto-skips whatever's
already done, so it's a single command rather than a hand-targeted
`--languages km` step. See "Automatic OOM retry/skip + auto-skip" above
for the exact command.

### Data actually evaluated (from `job-2854.out`, the `build_downstream_eval_data.py` run)

`--num-fewshot` was left at its default (`0`), so sib200/belebele are
scored 0-shot against their **full** test split, per language:

| task | docs/language | languages | total docs |
|---|---|---|---|
| sib200 | 204 | 8 | 1,632 |
| belebele | 895 | 8 (only `id` actually completed — see below) | 7,160 (target) |

For belebele, only `base/id` (895 docs, matching the current pinned eval
set) has valid predictions from the current run
(`data/downstream_predictions/base/belebele/id.jsonl`). `km`, `ms`, `th`,
`tl`, `vi` also have prediction files on disk, but each has **900** rows,
not 895 — they predate the current `build_downstream_eval_data.py` build
(a prior, differently-sized belebele pull) and don't correspond to the
currently pinned eval set or the current OOM-interrupted run. `my` and
`lo` have no belebele predictions for `base` at all. None of these stale
files are reflected in `data/evaluation_results.json` (belebele is absent
there entirely, for every model), so there's no risk of them silently
contaminating the results file — but they're worth deleting or
regenerating once belebele is rerun, so `data/downstream_predictions/`
doesn't mix two different eval-set vintages under the same task name.

mmlu is **not** the full 14,042-question test split — this run passed
`--mmlu-samples-per-subject 10`, so it's a 570-doc (10 questions × 57
subjects) evenly-sampled proxy rather than the complete benchmark. That's
enough for a fast forgetting *signal* but not a precise MMLU number —
worth re-running with a larger sample (or the full set, via neither flag)
if the ~7-8% forgetting gap above needs to be pinned down more precisely
than this pilot requires. 13.1's held-out perplexity uses the full fixed
`data/validation_set/{lang}.jsonl` for all 8 languages, unaffected by any
of the above (it doesn't go through `build_downstream_eval_data.py` at
all).

### 13.1 — held-out perplexity

| model | macro avg | worst language | rel. improvement vs base |
|---|---|---|---|
| base | 11.25 | 28.25 (tl) | — |
| best_weighted | 8.78 | 17.06 (tl) | +22.02% |
| random | 8.24 | 15.99 (tl) | **+26.75%** |
| clean_heuristic | 8.44 | 15.86 (tl) | +25.02% |
| equal_average | 8.87 | 17.25 (tl) | +21.22% |

All four CPT variants cut perplexity substantially vs. base, but
`best_weighted` is not the best of the four — `random` gives both the
largest macro reduction and the lowest worst-language perplexity.

### 13.2 — sib200 (0-shot topic classification)

| model | macro acc | worst language | rel. improvement vs base |
|---|---|---|---|
| base | 0.6097 | 0.2108 (my) | — |
| best_weighted | 0.6158 | 0.3284 (my) | +1.01% |
| random | 0.7426 | 0.5245 (my) | **+21.81%** |
| clean_heuristic | 0.6979 | 0.2745 (my) | +14.47% |
| equal_average | 0.4430 | 0.1275 (my) | -27.34% |

`random` again leads by a wide margin; `equal_average` actively hurts
sib200 accuracy despite having improved perplexity, showing 13.1 and 13.2
can rank the same baseline very differently. `my` (Burmese) is the
worst-performing language for every model, base included.

### mmlu (folded-in 13.3, forgetting/general-ability check)

| model | macro acc | worst subject | rel. improvement vs base |
|---|---|---|---|
| base | 0.6526 | 0.0000 (moral_scenarios) | — |
| best_weighted | 0.6053 | 0.1000 | -7.26% |
| random | 0.6000 | 0.2000 | -8.06% |
| clean_heuristic | 0.6018 | 0.1000 | -7.80% |
| equal_average | 0.6018 | 0.2000 | -7.80% |

All four CPT variants lose ~7-8% macro accuracy on English MMLU vs. base —
forgetting is roughly uniform across baselines regardless of which
weighting scheme produced the CPT data, so mmlu doesn't discriminate
between them the way perplexity/sib200 do.

### Reading against `pipeline.md` Section 14's claims

Claim 3 ("the best proxy-selected weighted data improves ... compared with
random, clean heuristic, and equal-average baselines") is **not supported**
by these results on either metric computed so far: `random` beats
`best_weighted` on both perplexity (26.75% vs. 22.02%) and sib200 (+21.81%
vs. +1.01%). belebele numbers (still missing) could change this picture,
but on the two benchmarks currently complete, `best_weighted` is the
weakest of the three real (non-random) baselines on sib200 and the weakest
of all four on perplexity.
