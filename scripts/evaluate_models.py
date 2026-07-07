"""
pipeline.md Section 13 - Evaluation.

Compares the untouched Original Qwen2.5-1.5B Base against one or more of
Section 7's final CPT baselines (LoRA adapters from run_final_cpt.py):
random, clean_heuristic, equal_average, best_weighted. Only
baselines that have actually been trained (i.e. whose adapter directory
exists under models/) are evaluated -- pass --baselines to pick which ones,
default is just best_weighted (the pilot's main method) to match the
original single-comparison behavior.

13.1 Held-out language modeling evaluation: per-language loss/perplexity on
     the fixed validation set (build_validation_set.py), same set used
     throughout the proxy search -- fully implemented and torch-based.

13.2 SEA downstream evaluation via lm-evaluation-harness:
     https://huggingface.co/datasets/Davlan/sib200 (topic classification)
     and belebele (MCQ). Shells out to the `lm_eval` CLI for each model and
     parses its results JSON.

     sib200 is wired up per-language: lm-evaluation-harness registers one
     subtask per FLORES-200 language code (e.g. `sib200_vie_Latn`), which
     happen to be exactly the same codes already used for
     `LANGUAGE_HF_CONFIGS` (FineWeb2 uses the same FLORES-200 taxonomy) --
     so `"sib200"` in `--tasks` expands to our 8 languages' subtasks
     automatically, and the per-language/macro-average/worst-language
     accuracy rollup pipeline.md's report asks for is computed directly
     from lm_eval's per-task results.

     belebele is passed through as-is (not expanded/rolled-up) since its
     exact per-language task-naming convention hasn't been verified against
     an installed lm-evaluation-harness -- double-check
     `lm_eval --tasks list` on the cluster and adjust if it isn't
     registered under the plain `"belebele"` group tag.

Usage:
    python3 scripts/evaluate_models.py --device cuda
    python3 scripts/evaluate_models.py --baselines random equal_average best_weighted --device cuda
    python3 scripts/evaluate_models.py --skip-downstream --device cuda  # 13.1 only
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sea_rater.baselines import BASELINE_CHOICES, MODEL_DIR_BY_BASELINE
from sea_rater.cpt import FINAL_MODEL_NAME, evaluate_loss, load_cpt_model_and_tokenizer, make_block_loader, pack_texts
from sea_rater.languages import LANGUAGE_HF_CONFIGS, LANGUAGES

REPO_ROOT = Path(__file__).resolve().parent.parent
VALIDATION_DIR = REPO_ROOT / "data" / "validation_set"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "evaluation_results.json"
DEFAULT_LM_EVAL_OUTPUT_DIR = REPO_ROOT / "data" / "lm_eval_raw"
DEFAULT_TASKS = ["sib200", "belebele"]

# lm-evaluation-harness's sib200 task group: one subtask per FLORES-200 code.
# https://huggingface.co/datasets/Davlan/sib200
SIB200_TASKS_BY_LANGUAGE = {lang: f"sib200_{hf_config}" for lang, hf_config in LANGUAGE_HF_CONFIGS.items()}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_lora_model(base_model_name, adapter_dir, device):
    from peft import PeftModel

    tokenizer, base_model = load_cpt_model_and_tokenizer(base_model_name, device=device)
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.to(device)
    return tokenizer, model


def run_held_out_eval(args):
    print("=== 13.1 Held-out language modeling evaluation ===")
    print(f"loading validation set for {len(LANGUAGES)} language(s) ...")
    texts_by_lang = {lang: [row["text"] for row in load_jsonl(VALIDATION_DIR / f"{lang}.jsonl")] for lang in LANGUAGES}

    print(f"[base] loading {args.model_name} ...")
    tokenizer_base, model_base = load_cpt_model_and_tokenizer(args.model_name, device=args.device)
    results = {"base": {}}
    for lang in tqdm(LANGUAGES, desc="base eval", unit="lang"):
        blocks_base = pack_texts(texts_by_lang[lang], tokenizer_base, args.seq_length, desc=f"base pack [{lang}]")
        loader_base = make_block_loader(blocks_base, args.eval_batch_size, shuffle=False)
        loss_base = evaluate_loss(model_base, loader_base, args.device, desc=f"base eval [{lang}]")
        results["base"][lang] = {"loss": loss_base, "perplexity": math.exp(loss_base)}
        tqdm.write(f"[base][{lang}] perplexity={results['base'][lang]['perplexity']:.2f}")

    ppls_base = [results["base"][lang]["perplexity"] for lang in LANGUAGES]
    results["base"]["macro_average_perplexity"] = sum(ppls_base) / len(ppls_base)
    results["base"]["worst_language_perplexity"] = max(ppls_base)

    del model_base
    if args.device == "cuda":
        torch.cuda.empty_cache()

    for baseline in tqdm(args.baselines, desc="baselines", unit="baseline"):
        adapter_dir = MODEL_DIR_BY_BASELINE[baseline]
        if not adapter_dir.exists():
            tqdm.write(f"[{baseline}] no trained adapter at {adapter_dir}, skipping "
                       f"(train it first: python scripts/run_final_cpt.py --baseline {baseline})")
            continue

        tqdm.write(f"[{baseline}] loading adapter from {adapter_dir} ...")
        tokenizer_cpt, model_cpt = load_lora_model(args.model_name, adapter_dir, args.device)
        results[baseline] = {}
        for lang in tqdm(LANGUAGES, desc=f"{baseline} eval", unit="lang", leave=False):
            blocks_cpt = pack_texts(texts_by_lang[lang], tokenizer_cpt, args.seq_length, desc=f"{baseline} pack [{lang}]")
            loader_cpt = make_block_loader(blocks_cpt, args.eval_batch_size, shuffle=False)
            loss_cpt = evaluate_loss(model_cpt, loader_cpt, args.device, desc=f"{baseline} eval [{lang}]")
            results[baseline][lang] = {"loss": loss_cpt, "perplexity": math.exp(loss_cpt)}
            tqdm.write(
                f"[{baseline}][{lang}] perplexity={results[baseline][lang]['perplexity']:.2f} "
                f"(base={results['base'][lang]['perplexity']:.2f})"
            )

        ppls_cpt = [results[baseline][lang]["perplexity"] for lang in LANGUAGES]
        results[baseline]["macro_average_perplexity"] = sum(ppls_cpt) / len(ppls_cpt)
        results[baseline]["worst_language_perplexity"] = max(ppls_cpt)

        macro_base = results["base"]["macro_average_perplexity"]
        macro_cpt = results[baseline]["macro_average_perplexity"]
        results[baseline]["relative_improvement_vs_base"] = (macro_base - macro_cpt) / macro_base
        tqdm.write(
            f"[{baseline}] macro perplexity={macro_cpt:.2f} (base={macro_base:.2f}, "
            f"relative improvement {results[baseline]['relative_improvement_vs_base']:.2%})"
        )

        del model_cpt
        if args.device == "cuda":
            torch.cuda.empty_cache()

    return results


def expand_tasks(tasks):
    """Replace the plain "sib200" group tag with its 8 per-language
    subtask names, so lm_eval only runs our languages instead of all ~200
    sib200 covers. Any other requested task (e.g. "belebele") passes
    through untouched."""
    expanded = []
    for task in tasks:
        if task == "sib200":
            expanded.extend(SIB200_TASKS_BY_LANGUAGE.values())
        else:
            expanded.append(task)
    return expanded


def summarize_sib200(lm_eval_results):
    """Pull accuracy out of lm_eval's per-task results for each of our 8
    languages' sib200_<flores_code> subtask and roll up macro-average /
    worst-language accuracy -- pipeline.md Section 13.2's report spec.
    Returns None if none of the expected per-task keys are present (e.g.
    sib200 wasn't in --tasks for this run)."""
    per_language = {}
    for lang, task_name in SIB200_TASKS_BY_LANGUAGE.items():
        task_results = lm_eval_results.get(task_name)
        if not task_results:
            continue
        acc = task_results.get("acc,none", task_results.get("acc"))
        if acc is not None:
            per_language[lang] = acc

    if not per_language:
        return None

    accs = list(per_language.values())
    return {
        "per_language_accuracy": per_language,
        "macro_average_accuracy": sum(accs) / len(accs),
        "worst_language_accuracy": min(accs),
    }


def run_lm_eval(model_args, output_path, tasks, device):
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", ",".join(expand_tasks(tasks)),
        "--device", device,
        "--output_path", str(output_path),
    ]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # lm_eval writes results under output_path/<model-name-slug>/results*.json
    result_files = list(output_path.rglob("results*.json"))
    if not result_files:
        raise FileNotFoundError(f"no lm_eval results json found under {output_path}")
    return json.loads(result_files[0].read_text(encoding="utf-8"))


def run_downstream_eval(args):
    print("\n=== 13.2 SEA downstream evaluation (lm-evaluation-harness) ===")
    args.lm_eval_output_dir.mkdir(parents=True, exist_ok=True)

    print("[base] running lm_eval ...")
    base_raw = run_lm_eval(
        f"pretrained={args.model_name}", args.lm_eval_output_dir / "base", args.tasks, args.device
    ).get("results", {})
    results = {"base": base_raw}
    sib200_base = summarize_sib200(base_raw)
    if sib200_base:
        results["base"]["sib200_summary"] = sib200_base
        print(
            f"[base] sib200 macro_average_accuracy={sib200_base['macro_average_accuracy']:.4f} "
            f"worst_language_accuracy={sib200_base['worst_language_accuracy']:.4f}"
        )

    for baseline in tqdm(args.baselines, desc="downstream eval", unit="baseline"):
        adapter_dir = MODEL_DIR_BY_BASELINE[baseline]
        if not adapter_dir.exists():
            tqdm.write(f"[{baseline}] no trained adapter at {adapter_dir}, skipping")
            continue
        tqdm.write(f"[{baseline}] running lm_eval ...")
        cpt_raw = run_lm_eval(
            f"pretrained={args.model_name},peft={adapter_dir}",
            args.lm_eval_output_dir / baseline,
            args.tasks,
            args.device,
        ).get("results", {})
        results[baseline] = cpt_raw
        sib200_cpt = summarize_sib200(cpt_raw)
        if sib200_cpt:
            results[baseline]["sib200_summary"] = sib200_cpt
            base_suffix = f" (base={sib200_base['macro_average_accuracy']:.4f})" if sib200_base else ""
            tqdm.write(
                f"[{baseline}] sib200 macro_average_accuracy={sib200_cpt['macro_average_accuracy']:.4f} "
                f"worst_language_accuracy={sib200_cpt['worst_language_accuracy']:.4f}{base_suffix}"
            )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baselines", nargs="+", choices=BASELINE_CHOICES, default=["best_weighted"],
        help="which trained final-CPT baselines to compare against the original base model",
    )
    parser.add_argument("--model-name", default=FINAL_MODEL_NAME)
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--lm-eval-output-dir", type=Path, default=DEFAULT_LM_EVAL_OUTPUT_DIR)
    parser.add_argument("--skip-downstream", action="store_true", help="only run 13.1, skip lm-evaluation-harness")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    results = {"held_out_lm_eval": run_held_out_eval(args)}

    if not args.skip_downstream:
        results["downstream_eval"] = run_downstream_eval(args)
    else:
        print("\n(skipping 13.2 downstream evaluation, --skip-downstream set)")

    args.output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote evaluation results -> {args.output_path}")


if __name__ == "__main__":
    main()
