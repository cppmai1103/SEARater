"""
pipeline.md Section 13 - Evaluation.

Compares the untouched Original Qwen2.5-1.5B Base against one or more of
Section 7's final CPT baselines (LoRA adapters from run_final_cpt.py):
random, clean_heuristic, equal_average, best_weighted. Only
baselines that have actually been trained (i.e. whose adapter directory
exists under models/) are evaluated -- pass --baselines to pick which ones,
default is just best_weighted (the pilot's main method) to match the
original single-comparison behavior.

Each model (base + every requested baseline) is loaded once and run through
both 13.1 and 13.2 before being freed, and results are written to disk after
every model finishes -- a crash partway through never discards an
already-evaluated model's numbers.

This script does zero dataset loading/preparation -- run
scripts/build_downstream_eval_data.py first (once; its output is reused
across every model/run). Splitting the two apart means this script never
touches the Hugging Face Hub or the `datasets` library, so re-running eval
(e.g. one baseline at a time, in a fresh process, for GPU-memory isolation)
never re-downloads or re-tokenizes anything, and the exact evaluated set is
pinned to disk rather than implicitly whatever the live HF dataset returns
at the moment a job happens to run.

13.1 Held-out language modeling evaluation: per-language loss/perplexity on
     the fixed validation set (build_validation_set.py), same set used
     throughout the proxy search.

13.2 SEA downstream evaluation: 0-shot multiple-choice accuracy on
     Davlan/sib200 (7-way topic classification, per-language) and
     facebook/belebele (4-way reading comprehension, per-language), plus
     13.3 folded in here rather than kept as a separate unimplemented
     section: cais/mmlu (4-way, English-only, per-subject) as a general-
     ability check -- "does continued pretraining on SEA data hurt general
     knowledge?" -- not part of the per-language SEA rollup, since MMLU has
     no SEA-language coverage to begin with. All three scored directly
     against the loaded model with our own log-likelihood scoring
     (score_continuations below) -- no lm-evaluation-harness dependency.
     That was dropped because sib200 isn't a registered lm-eval task at all
     in the installed version (verified: no lm_eval/tasks/sib200/ in the
     package, and a real cluster run reported every sib200_<code> as "not
     found"), and shelling out to a separate `lm_eval` CLI process per
     model/baseline was slower and harder to debug than scoring in-process,
     especially since 13.1 already loads each model once anyway.

     All three are framed as real lettered multiple-choice (see
     build_downstream_eval_data.py's format_mcq_context: every choice is
     listed inline under a letter, "A. ...\nB. ...\nSelect the one correct
     letter for the answer.\nAnswer:", and only the single letters are
     scored as continuations -- plain continuation-scoring has a length
     bias that lettered choices remove, since every candidate becomes one
     token). Loaded here as pre-built local JSONL
     (data/downstream_eval_data/{sib200,belebele}/{lang}.jsonl,
     data/downstream_eval_data/mmlu.jsonl tagged with a "subject" field) --
     see build_downstream_eval_data.py for the k-shot/dataset-schema/
     language-config details, none of which this script needs to know
     about anymore.

     Two metrics per bucket (language for sib200/belebele, academic subject
     for mmlu -- BUCKET_LABEL_BY_TASK), both rolled up to macro-average/
     worst-bucket: `accuracy` (argmax over raw summed log-likelihood,
     matching lm-eval's plain "acc") and `token_normalized_prob_correct`
     (`exp(sum_log_prob_of_gold_choice / num_tokens_in_gold_choice)`,
     averaged over docs) -- a continuous confidence-in-the-correct-answer
     score that still moves when a model gets closer to right without
     flipping its top-1 pick, unlike accuracy's 0/1. Every doc's prediction
     (predicted vs. gold choice, per-choice log-likelihoods, whether it was
     correct) is also written to
     `<predictions_dir>/<model_tag>/<task>/<bucket>.jsonl` for error
     analysis.

Usage:
    python3 scripts/build_downstream_eval_data.py   # once, before any of the below
    python3 scripts/evaluate_models.py --device cuda
    python3 scripts/evaluate_models.py --baselines random equal_average best_weighted --device cuda
    python3 scripts/evaluate_models.py --task perplexity --device cuda  # 13.1 only, no local eval data needed
    python3 scripts/evaluate_models.py --task mmlu --device cuda        # one task only, fresh process

    # one command handles the whole remaining belebele run: per-bucket OOM retry/skip
    # (evaluate_bucket_with_retry) plus default auto-skip-what's-already-done means this
    # neither crashes on whichever language OOMs nor redoes languages already scored --
    # just re-run this same command again later if anything got skipped this time:
    python3 scripts/evaluate_models.py --task belebele --eval-batch-size 8 --downstream-batch-size 8 \
        --baselines best_weighted random clean_heuristic equal_average --device cuda
    # --languages restricts to specific codes only, if you want to scope a run manually instead:
    python3 scripts/evaluate_models.py --task belebele --languages km lo --device cuda
"""

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sea_rater.baselines import BASELINE_CHOICES, MODEL_DIR_BY_BASELINE
from sea_rater.cpt import FINAL_MODEL_NAME, evaluate_loss, load_cpt_model_and_tokenizer, make_block_loader, pack_texts
from sea_rater.languages import LANGUAGES

REPO_ROOT = Path(__file__).resolve().parent.parent
VALIDATION_DIR = REPO_ROOT / "data" / "validation_set"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "evaluation_results.json"
DEFAULT_PREDICTIONS_DIR = REPO_ROOT / "data" / "downstream_predictions"
DEFAULT_EVAL_DATA_DIR = REPO_ROOT / "data" / "downstream_eval_data"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_lora_model(base_model_name, adapter_dir, device):
    from peft import PeftModel

    tokenizer, base_model = load_cpt_model_and_tokenizer(base_model_name, device=device)
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.to(device)
    return tokenizer, model


def score_continuations(model, tokenizer, contexts, continuations, device, batch_size, desc):
    """Sum log P(continuation | context) and the continuation's token count,
    for each parallel (context, continuation) pair -- the same 0-shot
    scoring lm-evaluation-harness uses for output_type: multiple_choice
    tasks (a "loglikelihood" request). Batched over sequences (not choices,
    since each row's continuation length differs and can't share one
    forward pass's positions). Returns a list of (sum_log_prob, num_tokens)
    pairs, one per (context, continuation) pair -- the token count is kept
    around so callers can length-normalize (raw sum log-likelihood favors
    shorter continuations).

    Pairs are sorted by token length before batching (like
    lm-evaluation-harness does internally), not processed in the caller's
    original order: without this, each batch's padding target (and so its
    logits tensor's memory footprint) swings unpredictably from batch to
    batch depending on which specific docs happen to land together, which
    -- confirmed by a real crash mid-bucket, deep into a long-running
    per-language loop, with a large and growing "reserved by PyTorch but
    unallocated" figure in the OOM message -- fragments PyTorch's caching
    allocator badly enough to eventually fail an allocation that would fit
    in the nominally-free memory. Sorting means allocation sizes change
    smoothly across the run instead of jumping randomly, and also cuts
    padding waste since same-length sequences batch together. Results are
    unsorted back to the caller's original order before returning."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    n = len(contexts)
    context_ids_all = [tokenizer(c, add_special_tokens=False).input_ids for c in contexts]
    full_ids_all = [
        c_ids + tokenizer(cont, add_special_tokens=False).input_ids
        for c_ids, cont in zip(context_ids_all, continuations)
    ]
    order = sorted(range(n), key=lambda i: len(full_ids_all[i]))

    results = [None] * n
    for batch_start in tqdm(range(0, n, batch_size), desc=desc, unit="batch", leave=False):
        batch_order = order[batch_start : batch_start + batch_size]
        batch_full_ids = [full_ids_all[i] for i in batch_order]
        batch_context_ids = [context_ids_all[i] for i in batch_order]

        max_len = max(len(ids) for ids in batch_full_ids)
        input_ids = torch.full((len(batch_full_ids), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch_full_ids), max_len), dtype=torch.long)
        for row, ids in enumerate(batch_full_ids):
            input_ids[row, : len(ids)] = torch.tensor(ids)
            attention_mask[row, : len(ids)] = 1
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask=attention_mask).logits.float()
        log_probs = torch.log_softmax(logits, dim=-1)

        for row, (doc_index, c_ids) in enumerate(zip(batch_order, batch_context_ids)):
            cont_ids = batch_full_ids[row][len(c_ids) :]
            start_pos = len(c_ids) - 1
            token_log_probs = log_probs[row, start_pos : start_pos + len(cont_ids), :]
            cont_ids_tensor = torch.tensor(cont_ids, device=device)
            picked = token_log_probs.gather(1, cont_ids_tensor.unsqueeze(-1)).squeeze(-1)
            results[doc_index] = (picked.sum().item(), len(cont_ids))
    return results


def evaluate_multiple_choice(model, tokenizer, docs, device, batch_size, desc):
    """docs: list of {"context", "choices", "choice_texts", "gold"}, where
    `context` already lists every choice inline under a letter
    (format_mcq_context) and `choices` are the single-letter continuations
    to score (e.g. ["A", "B", "C", "D"]) -- `choice_texts` are the original
    answer strings, kept only for readable predictions output. Scores every
    (context, " " + letter) pair, picks the letter with the highest raw
    summed log-likelihood per doc (accuracy -- matches lm-eval's plain
    "acc"), and separately computes the *gold* letter's token-normalized
    probability `exp(sum_log_prob / num_tokens)` -- a continuous confidence
    score for specifically the correct answer, independent of whether it
    won the argmax. Unlike accuracy (which only ever reads 0 or 1), this
    still moves when a model's belief in the correct answer changes without
    flipping its top-1 pick, so it can show a CPT model "getting warmer"
    even where accuracy alone looks flat.

    Returns (accuracy, mean_token_normalized_prob_correct, predictions) --
    predictions is one dict per doc, for saving to disk."""
    contexts, continuations, doc_spans = [], [], []
    for doc in docs:
        start = len(contexts)
        for letter in doc["choices"]:
            contexts.append(doc["context"])
            continuations.append(" " + letter)
        doc_spans.append((start, len(contexts)))

    scored = score_continuations(model, tokenizer, contexts, continuations, device, batch_size, desc)

    correct = 0
    token_normalized_probs = []
    predictions = []
    for doc, (start, end) in zip(docs, doc_spans):
        doc_scores = scored[start:end]  # [(sum_log_prob, num_tokens), ...] one per letter
        log_likelihoods = [s for s, _ in doc_scores]
        predicted = max(range(len(log_likelihoods)), key=lambda i: log_likelihoods[i])
        is_correct = predicted == doc["gold"]
        correct += int(is_correct)

        gold_sum_log_prob, gold_num_tokens = doc_scores[doc["gold"]]
        token_normalized_prob_correct = math.exp(gold_sum_log_prob / gold_num_tokens)
        token_normalized_probs.append(token_normalized_prob_correct)

        predictions.append(
            {
                "context": doc["context"],
                "choices": doc["choice_texts"],
                "gold": doc["gold"],
                "gold_letter": doc["choices"][doc["gold"]],
                "gold_choice": doc["choice_texts"][doc["gold"]],
                "predicted": predicted,
                "predicted_letter": doc["choices"][predicted],
                "predicted_choice": doc["choice_texts"][predicted],
                "correct": is_correct,
                "choice_log_likelihoods": log_likelihoods,
                "token_normalized_prob_correct": token_normalized_prob_correct,
            }
        )

    accuracy = correct / len(docs)
    mean_token_normalized_prob_correct = sum(token_normalized_probs) / len(token_normalized_probs)
    return accuracy, mean_token_normalized_prob_correct, predictions


def load_local_docs_per_bucket(task_dir, languages=None):
    """<task_dir>/<bucket>.jsonl per file -- sib200/belebele's layout, as
    written by build_downstream_eval_data.py's save_docs. `languages`
    (from --languages), if given, restricts to just those language codes --
    e.g. re-running belebele after an OOM crash on one specific language
    without reloading/rescoring the others."""
    docs = {path.stem: load_jsonl(path) for path in sorted(task_dir.glob("*.jsonl"))}
    if languages is not None:
        docs = {lang: rows for lang, rows in docs.items() if lang in languages}
    return docs


def load_local_docs_combined(path, bucket_label):
    """One file, every row tagged with `bucket_label` -- mmlu's layout, as
    written by build_downstream_eval_data.py's save_docs_combined."""
    docs_by_bucket = {}
    for row in load_jsonl(path):
        bucket = row.pop(bucket_label)
        docs_by_bucket.setdefault(bucket, []).append(row)
    return docs_by_bucket


DOWNSTREAM_TASKS = ["sib200", "belebele", "mmlu"]


def load_downstream_docs(eval_data_dir, tasks, languages=None):
    """Loads only the requested `tasks` (subset of DOWNSTREAM_TASKS) --
    when --task scopes a run to just one downstream task (or to
    "perplexity", none), there's no reason to read the other tasks' local
    JSONL off disk at all. `languages` restricts sib200/belebele (bucketed
    by language) to just those codes; ignored for mmlu (bucketed by
    subject, not language)."""
    docs = {}
    for task_name in tasks:
        if task_name == "sib200":
            path = eval_data_dir / "sib200"
            if not path.exists():
                raise FileNotFoundError(f"{path} not found -- run scripts/build_downstream_eval_data.py first")
            docs["sib200"] = load_local_docs_per_bucket(path, languages)
        elif task_name == "belebele":
            path = eval_data_dir / "belebele"
            if not path.exists():
                raise FileNotFoundError(f"{path} not found -- run scripts/build_downstream_eval_data.py first")
            docs["belebele"] = load_local_docs_per_bucket(path, languages)
        elif task_name == "mmlu":
            path = eval_data_dir / "mmlu.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"{path} not found -- run scripts/build_downstream_eval_data.py first")
            docs["mmlu"] = load_local_docs_combined(path, "subject")
    if docs:
        print(f"loading downstream eval data ({', '.join(docs)}) from {eval_data_dir} ...")
    return docs


def save_predictions(predictions_dir, model_tag, task_name, bucket, predictions):
    """One file per bucket: <predictions_dir>/<model_tag>/<task_name>/<bucket>.jsonl"""
    out_dir = predictions_dir / model_tag / task_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{bucket}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out_path


def save_predictions_combined(predictions_dir, model_tag, task_name, predictions):
    """One file for the whole task: <predictions_dir>/<model_tag>/<task_name>.jsonl
    -- every row still carries its bucket (e.g. "subject") as a field, added
    by rollup_downstream_task, so the combined file stays filterable/
    groupable by bucket despite not being split into separate files."""
    out_dir = predictions_dir / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task_name}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out_path


# sib200/belebele are bucketed by language (2-letter code); mmlu -- a single
# English general-ability check, not part of the per-language SEA rollup --
# is bucketed by academic subject instead. Both use the exact same
# rollup_downstream_task below; only the JSON key names ("per_language_..."
# vs. "per_subject_...") change, via this lookup.
BUCKET_LABEL_BY_TASK = {"sib200": "language", "belebele": "language", "mmlu": "subject"}

# sib200/belebele: one predictions file per language (8 files, each a
# manageable size). mmlu: 57 subjects x a handful of samples each is small
# enough, and more useful, as one combined file -- each row tagged with a
# "subject" field (via BUCKET_LABEL_BY_TASK) rather than split across 57
# separate per-subject files.
COMBINE_PREDICTIONS_BY_TASK = {"sib200": False, "belebele": False, "mmlu": True}


def evaluate_bucket_with_retry(model, tokenizer, docs, device, batch_size, desc):
    """Wraps evaluate_multiple_choice with an OOM retry: on
    torch.OutOfMemoryError, clears the cache and retries the same bucket at
    half the batch size (down to a floor of 1) instead of letting one
    over-long bucket (e.g. belebele's long passages) crash the whole run.
    Returns None if it still OOMs at batch_size=1, so the caller can skip
    just this bucket and keep going with the rest -- confirmed necessary by
    two real crashes (job-2887, job-2897) that both took down the entire
    process on belebele's `km`, discarding every language after it even
    though most of them fit fine on their own."""
    current_bs = batch_size
    while True:
        try:
            return evaluate_multiple_choice(model, tokenizer, docs, device, current_bs, desc)
        except torch.OutOfMemoryError as e:
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            if current_bs <= 1:
                tqdm.write(f"{desc}: still OOM at batch_size=1, skipping this bucket for now ({e})")
                return None
            current_bs = max(1, current_bs // 2)
            tqdm.write(f"{desc}: OOM, retrying at batch_size={current_bs}")


def rollup_downstream_task(model, tokenizer, device, batch_size, model_tag, task_name, docs_by_bucket, predictions_dir):
    """0-shot accuracy + token-normalized correct-answer probability for one
    task, for one already-loaded model, bucketed by whatever
    BUCKET_LABEL_BY_TASK says (language for sib200/belebele, subject for
    mmlu) and rolled up to macro-average/worst-bucket. Every doc's
    prediction is tagged with its bucket and written to disk (one file per
    bucket, or one combined file per task -- COMBINE_PREDICTIONS_BY_TASK)
    for error analysis -- accuracy alone doesn't show *which* docs a model
    got wrong or how confident it was.

    Returns None if every requested bucket OOM'd even at batch_size=1 (so
    the caller has nothing to merge); a subset of buckets can also be
    missing from a non-None result if only some OOM'd -- those stay
    unevaluated and get picked up by a later run's auto-skip logic, same as
    a bucket nobody asked for this run."""
    bucket_label = BUCKET_LABEL_BY_TASK[task_name]
    combine = COMBINE_PREDICTIONS_BY_TASK[task_name]
    per_bucket_accuracy = {}
    per_bucket_token_normalized_prob = {}
    combined_predictions = []
    skipped_buckets = []
    for bucket, docs in docs_by_bucket.items():
        result = evaluate_bucket_with_retry(
            model, tokenizer, docs, device, batch_size, f"{model_tag} {task_name} [{bucket}]"
        )
        if result is None:
            skipped_buckets.append(bucket)
            continue
        acc, token_normalized_prob, predictions = result
        for row in predictions:
            row[bucket_label] = bucket
        per_bucket_accuracy[bucket] = acc
        per_bucket_token_normalized_prob[bucket] = token_normalized_prob

        if combine:
            combined_predictions.extend(predictions)
            tqdm.write(
                f"[{model_tag}] {task_name} [{bucket}] accuracy={acc:.4f} "
                f"token_normalized_prob_correct={token_normalized_prob:.4f}"
            )
        else:
            out_path = save_predictions(predictions_dir, model_tag, task_name, bucket, predictions)
            tqdm.write(
                f"[{model_tag}] {task_name} [{bucket}] accuracy={acc:.4f} "
                f"token_normalized_prob_correct={token_normalized_prob:.4f} -> {out_path}"
            )

        # Same fix as run_proxy_cpt.py's run_one_weight: CUDA tensors from
        # score_continuations' forward passes can form reference cycles that
        # plain refcounting won't clear even under torch.no_grad(), so
        # gc.collect() has to run before empty_cache() has anything to
        # release. Without this, memory creeps up across buckets (and across
        # sib200 -> belebele -> mmlu, and across models) until a later one
        # OOMs even though each individual bucket fits comfortably on its
        # own -- confirmed by a real crash 6 languages into belebele, deep
        # into a batch, with far more "allocated by PyTorch" memory in use
        # than a single 1.5B-model forward pass should ever need.
        if device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

    if combine:
        out_path = save_predictions_combined(predictions_dir, model_tag, task_name, combined_predictions)
        tqdm.write(f"[{model_tag}] {task_name} predictions ({len(combined_predictions)} rows) -> {out_path}")

    if skipped_buckets:
        tqdm.write(f"[{model_tag}] {task_name}: skipped {sorted(skipped_buckets)} (OOM even at batch_size=1) "
                    f"-- re-run this exact command later (auto-skip will only retry these)")

    if not per_bucket_accuracy:
        return None

    accs = list(per_bucket_accuracy.values())
    token_normalized_probs = list(per_bucket_token_normalized_prob.values())
    result = {
        f"per_{bucket_label}_accuracy": per_bucket_accuracy,
        "macro_average_accuracy": sum(accs) / len(accs),
        f"worst_{bucket_label}_accuracy": min(accs),
        f"per_{bucket_label}_token_normalized_prob_correct": per_bucket_token_normalized_prob,
        "macro_average_token_normalized_prob_correct": sum(token_normalized_probs) / len(token_normalized_probs),
        f"worst_{bucket_label}_token_normalized_prob_correct": min(token_normalized_probs),
    }
    tqdm.write(
        f"[{model_tag}] {task_name} macro_average_accuracy={result['macro_average_accuracy']:.4f} "
        f"worst_{bucket_label}_accuracy={result[f'worst_{bucket_label}_accuracy']:.4f} "
        f"macro_average_token_normalized_prob_correct={result['macro_average_token_normalized_prob_correct']:.4f}"
    )
    return result


def run_downstream_eval(model, tokenizer, device, batch_size, model_tag, downstream_docs, predictions_dir):
    """0-shot sib200 + belebele + mmlu accuracy (and token-normalized
    correct-answer probability) for one already-loaded model. A task is
    dropped entirely from the result (rather than included as None) if
    every one of its buckets OOM'd even at batch_size=1 -- there's nothing
    for merge_model_result to merge in that case, and next run's auto-skip
    will retry it since it's still missing from --output-path."""
    results = {}
    for task_name, docs_by_bucket in downstream_docs.items():
        result = rollup_downstream_task(
            model, tokenizer, device, batch_size, model_tag, task_name, docs_by_bucket, predictions_dir
        )
        if result is not None:
            results[task_name] = result
    return results


def evaluate_one_model(model, tokenizer, device, texts_by_lang, args, model_tag, downstream_docs, run_perplexity):
    """Computes only what this run's --task actually asked for:
    `run_perplexity` gates 13.1, `downstream_docs` (possibly empty, never
    None) gates 13.2 -- each --task invocation only pays for the one thing
    it's scoped to, both in compute and in GPU-memory-isolation terms (a
    fresh process per --task never accumulates memory across tasks)."""
    # Explicit rather than relying on evaluate_loss's internal model.eval()
    # call from the 13.1 loop below happening to run first: LoRA adapters
    # here use lora_dropout=0.05 (nonzero), so if 13.2's MCQ scoring ever
    # ran without eval mode set (e.g. with --task sib200 and no 13.1 in the
    # same process), dropout would still be active and make
    # accuracy/token_normalized_prob_correct non-deterministic per run.
    model.eval()
    result = {}

    if run_perplexity:
        result["held_out_lm_eval"] = {}
        for lang in tqdm(LANGUAGES, desc=f"{model_tag} perplexity", unit="lang"):
            blocks = pack_texts(texts_by_lang[lang], tokenizer, args.seq_length, desc=f"{model_tag} pack [{lang}]")
            loader = make_block_loader(blocks, args.eval_batch_size, shuffle=False)
            loss = evaluate_loss(model, loader, device, desc=f"{model_tag} eval [{lang}]")
            result["held_out_lm_eval"][lang] = {"loss": loss, "perplexity": math.exp(loss)}
            tqdm.write(f"[{model_tag}][{lang}] perplexity={math.exp(loss):.2f}")

        ppls = [result["held_out_lm_eval"][lang]["perplexity"] for lang in LANGUAGES]
        result["held_out_lm_eval"]["macro_average_perplexity"] = sum(ppls) / len(ppls)
        result["held_out_lm_eval"]["worst_language_perplexity"] = max(ppls)
        tqdm.write(f"[{model_tag}] macro perplexity={result['held_out_lm_eval']['macro_average_perplexity']:.2f}")

    if downstream_docs:
        result["downstream_eval"] = run_downstream_eval(
            model, tokenizer, device, args.downstream_batch_size, model_tag, downstream_docs, args.predictions_dir
        )

    return result


def add_relative_improvement(results, baseline):
    """Recomputes relative_improvement_vs_base for whatever `baseline` and
    `"base"` currently both have in `results` -- not just what this
    specific run just computed. With --task splitting each run to one task,
    "base" and `baseline` for a given task can land in the results file
    from two entirely different invocations (in either order), so this is
    deliberately opportunistic/idempotent rather than assuming both sides
    were just freshly computed together in this same process."""
    base = results.get("base", {})
    cpt = results.get(baseline, {})

    if "held_out_lm_eval" in base and "held_out_lm_eval" in cpt:
        macro_base = base["held_out_lm_eval"]["macro_average_perplexity"]
        macro_cpt = cpt["held_out_lm_eval"]["macro_average_perplexity"]
        cpt["held_out_lm_eval"]["relative_improvement_vs_base"] = (macro_base - macro_cpt) / macro_base
        tqdm.write(
            f"[{baseline}] macro perplexity={macro_cpt:.2f} (base={macro_base:.2f}, "
            f"relative improvement {cpt['held_out_lm_eval']['relative_improvement_vs_base']:.2%})"
        )

    if "downstream_eval" in cpt and "downstream_eval" in base:
        for task_name in cpt["downstream_eval"]:
            if task_name not in base["downstream_eval"]:
                continue
            task_cpt = cpt["downstream_eval"][task_name]
            task_base = base["downstream_eval"][task_name]

            acc_base = task_base["macro_average_accuracy"]
            acc_cpt = task_cpt["macro_average_accuracy"]
            task_cpt["relative_improvement_vs_base"] = (acc_cpt - acc_base) / acc_base

            prob_base = task_base["macro_average_token_normalized_prob_correct"]
            prob_cpt = task_cpt["macro_average_token_normalized_prob_correct"]
            task_cpt["token_normalized_prob_correct_relative_improvement_vs_base"] = (prob_cpt - prob_base) / prob_base

            tqdm.write(
                f"[{baseline}] {task_name} macro_average_accuracy={acc_cpt:.4f} (base={acc_base:.4f}, "
                f"relative improvement {task_cpt['relative_improvement_vs_base']:.2%}) "
                f"macro_average_token_normalized_prob_correct={prob_cpt:.4f} (base={prob_base:.4f}, "
                f"relative improvement {task_cpt['token_normalized_prob_correct_relative_improvement_vs_base']:.2%})"
            )


def load_existing_results(output_path):
    if output_path.exists():
        return json.loads(output_path.read_text(encoding="utf-8"))
    return {}


def filter_unevaluated_buckets(downstream_docs, existing_model_result, model_tag, force):
    """Drops buckets (languages for sib200/belebele, subjects for mmlu)
    already scored for this specific model in a prior run -- so re-running
    e.g. --task belebele after an OOM crash only pays for whatever this
    model tag hasn't finished yet (skips `id`, which already succeeded for
    `base`), instead of redoing every bucket from scratch every invocation.
    Different model tags can have different already-done sets (e.g. `base`
    has `id` done, a baseline may have none yet), so this is applied once
    per model right before that model is evaluated, not globally. Pass
    --force-recompute to disable and always redo everything (e.g. after
    retraining an adapter, when old numbers are stale)."""
    if force:
        return downstream_docs
    existing_downstream = existing_model_result.get("downstream_eval", {})
    filtered = {}
    for task_name, docs_by_bucket in downstream_docs.items():
        bucket_label = BUCKET_LABEL_BY_TASK[task_name]
        already_done = set(existing_downstream.get(task_name, {}).get(f"per_{bucket_label}_accuracy", {}))
        remaining = {bucket: docs for bucket, docs in docs_by_bucket.items() if bucket not in already_done}
        skipped = sorted(already_done & docs_by_bucket.keys())
        if skipped:
            tqdm.write(f"[{model_tag}] {task_name}: skipping already-evaluated {bucket_label}(s) {skipped} "
                       f"(pass --force-recompute to redo)")
        if remaining:
            filtered[task_name] = remaining
    return filtered


def merge_downstream_task_result(existing_task, new_task, bucket_label):
    """Unions per-bucket entries from `new_task` into `existing_task` (both
    in rollup_downstream_task's result shape) and recomputes macro/worst
    from the merged per-bucket dicts, instead of the old plain-replace
    behavior. Needed once --languages can scope a belebele/sib200 run to a
    handful of buckets at a time (e.g. resuming after an OOM crash one
    language at a time): without this, re-running belebele with just
    --languages km would discard every other language (like id) an earlier
    run already finished for the same model, since the old code did
    `existing.update(new)` at the whole-task level."""
    if existing_task is None:
        return new_task
    merged = dict(existing_task)
    acc_key, prob_key, worst_acc_key, worst_prob_key = (
        f"per_{bucket_label}_accuracy",
        f"per_{bucket_label}_token_normalized_prob_correct",
        f"worst_{bucket_label}_accuracy",
        f"worst_{bucket_label}_token_normalized_prob_correct",
    )
    merged[acc_key] = {**existing_task[acc_key], **new_task[acc_key]}
    merged[prob_key] = {**existing_task[prob_key], **new_task[prob_key]}
    accs = list(merged[acc_key].values())
    probs = list(merged[prob_key].values())
    merged["macro_average_accuracy"] = sum(accs) / len(accs)
    merged[worst_acc_key] = min(accs)
    merged["macro_average_token_normalized_prob_correct"] = sum(probs) / len(probs)
    merged[worst_prob_key] = min(probs)
    return merged


def merge_model_result(results, model_tag, new_result):
    """Merges `new_result` (whatever subset of {"held_out_lm_eval",
    "downstream_eval": {task: {...}}} this run's --task/--languages
    computed) into results[model_tag], preserving keys already there from
    earlier partial runs (a different --task/--languages, possibly a
    different process entirely) instead of overwriting the whole model
    entry -- and, within a downstream task, merging per-bucket (language/
    subject) rather than replacing the whole task, so a --languages-scoped
    run only ever adds buckets, never discards ones already done."""
    existing = results.setdefault(model_tag, {})
    if "held_out_lm_eval" in new_result:
        existing["held_out_lm_eval"] = new_result["held_out_lm_eval"]
    if "downstream_eval" in new_result:
        existing_downstream = existing.setdefault("downstream_eval", {})
        for task_name, task_result in new_result["downstream_eval"].items():
            bucket_label = BUCKET_LABEL_BY_TASK[task_name]
            existing_downstream[task_name] = merge_downstream_task_result(
                existing_downstream.get(task_name), task_result, bucket_label
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baselines", nargs="+", choices=BASELINE_CHOICES, default=["best_weighted"],
        help="which trained final-CPT baselines to compare against the original base model",
    )
    parser.add_argument("--model-name", default=FINAL_MODEL_NAME)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--downstream-batch-size", type=int, default=16)
    parser.add_argument(
        "--task", choices=["all", "perplexity"] + DOWNSTREAM_TASKS, default="all",
        help="run just one task per invocation ('perplexity' for 13.1, or one of sib200/belebele/mmlu for "
             "13.2) instead of everything in one process -- e.g. 4 separate invocations, one per task, each "
             "a fresh process, so a crash or memory buildup in one task (see docs/09_evaluation.md's OOM "
             "section) can never affect another. Results merge into --output-path rather than overwriting, "
             "so 'base' and a baseline for the same task can land in the file from two different runs, in "
             "either order, and relative_improvement_vs_base is recomputed opportunistically once both exist.",
    )
    parser.add_argument(
        "--eval-data-dir", type=Path, default=DEFAULT_EVAL_DATA_DIR,
        help="pre-built downstream eval data from scripts/build_downstream_eval_data.py "
             "(run that first -- this script does no dataset loading of its own)",
    )
    parser.add_argument(
        "--languages", nargs="+", default=None,
        help="restrict sib200/belebele to just these language codes (e.g. --languages km) -- lets a belebele "
             "re-run after an OOM crash target one specific language first instead of all 8. Ignored for mmlu "
             "(bucketed by subject, not language).",
    )
    parser.add_argument(
        "--force-recompute", action="store_true",
        help="by default, a (model, task, language/subject) combo already present in --output-path is skipped "
             "-- e.g. re-running --task belebele only scores whatever didn't finish last time. Pass this to "
             "always recompute every requested bucket regardless of what's already on disk (e.g. after "
             "retraining an adapter, when old numbers are stale).",
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS_DIR,
        help="where per-doc downstream-eval predictions are written: one <model>/<task>/<bucket>.jsonl per "
             "language for sib200/belebele, one combined <model>/mmlu.jsonl (tagged with a \"subject\" field) for mmlu",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_perplexity = args.task in ("all", "perplexity")
    downstream_tasks = DOWNSTREAM_TASKS if args.task == "all" else [args.task] if args.task != "perplexity" else []

    texts_by_lang = {}
    if run_perplexity:
        print(f"loading validation set for {len(LANGUAGES)} language(s) ...")
        texts_by_lang = {lang: [row["text"] for row in load_jsonl(VALIDATION_DIR / f"{lang}.jsonl")] for lang in LANGUAGES}
    downstream_docs = load_downstream_docs(args.eval_data_dir, downstream_tasks, args.languages)

    results = load_existing_results(args.output_path)

    print(f"[base] loading {args.model_name} ...")
    tokenizer, model = load_cpt_model_and_tokenizer(args.model_name, device=args.device)
    base_downstream_docs = filter_unevaluated_buckets(
        downstream_docs, results.get("base", {}), "base", args.force_recompute
    )
    base_result = evaluate_one_model(
        model, tokenizer, args.device, texts_by_lang, args, "base", base_downstream_docs, run_perplexity
    )
    merge_model_result(results, "base", base_result)
    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()
    args.output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote evaluation results -> {args.output_path}")

    for baseline in tqdm(args.baselines, desc="baselines", unit="baseline"):
        adapter_dir = MODEL_DIR_BY_BASELINE[baseline]
        if not adapter_dir.exists():
            tqdm.write(f"[{baseline}] no trained adapter at {adapter_dir}, skipping "
                       f"(train it first: python scripts/run_final_cpt.py --baseline {baseline})")
            continue

        tqdm.write(f"[{baseline}] loading adapter from {adapter_dir} ...")
        tokenizer, model = load_lora_model(args.model_name, adapter_dir, args.device)
        baseline_downstream_docs = filter_unevaluated_buckets(
            downstream_docs, results.get(baseline, {}), baseline, args.force_recompute
        )
        cpt_result = evaluate_one_model(
            model, tokenizer, args.device, texts_by_lang, args, baseline, baseline_downstream_docs, run_perplexity
        )
        merge_model_result(results, baseline, cpt_result)
        add_relative_improvement(results, baseline)

        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()
        args.output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote evaluation results -> {args.output_path}")


if __name__ == "__main__":
    main()
