"""
pipeline.md Section 13 - Evaluation.

Compares the two final models:
    Original Qwen2.5-1.5B Base
    Qwen2.5-1.5B-CPT-BestWeighted (LoRA adapter from run_final_cpt.py)

13.1 Held-out language modeling evaluation: per-language loss/perplexity on
     the fixed validation set (build_validation_set.py), same set used
     throughout the proxy search -- fully implemented and torch-based.

13.2 SEA downstream evaluation via lm-evaluation-harness (sib200 topic
     classification, belebele MCQ): shells out to the `lm_eval` CLI for
     each model and parses its results JSON. This part is best-effort --
     exact task names/configs in lm-evaluation-harness can differ by
     installed version, so double-check `lm_eval --tasks list` on the
     cluster and adjust --tasks if sib200/belebele aren't registered under
     those exact names.

Usage:
    python3 scripts/evaluate_models.py --device cuda
    python3 scripts/evaluate_models.py --skip-downstream --device cuda  # 13.1 only
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sea_rater.cpt import FINAL_MODEL_NAME, evaluate_loss, load_cpt_model_and_tokenizer, make_block_loader, pack_texts

REPO_ROOT = Path(__file__).resolve().parent.parent
VALIDATION_DIR = REPO_ROOT / "data" / "validation_set"
DEFAULT_ADAPTER_DIR = REPO_ROOT / "models" / "qwen1.5b_cpt_bestweighted"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "evaluation_results.json"
DEFAULT_LM_EVAL_OUTPUT_DIR = REPO_ROOT / "data" / "lm_eval_raw"

LANGUAGES = ["vi", "id", "th", "km"]
DEFAULT_TASKS = ["sib200", "belebele"]


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
    tokenizer_base, model_base = load_cpt_model_and_tokenizer(args.model_name, device=args.device)
    tokenizer_cpt, model_cpt = load_lora_model(args.model_name, args.adapter_dir, args.device)

    results = {"base": {}, "cpt_best_weighted": {}}
    for lang in LANGUAGES:
        rows = load_jsonl(VALIDATION_DIR / f"{lang}.jsonl")
        texts = [row["text"] for row in rows]

        blocks_base = pack_texts(texts, tokenizer_base, args.seq_length)
        loader_base = make_block_loader(blocks_base, args.eval_batch_size, shuffle=False)
        loss_base = evaluate_loss(model_base, loader_base, args.device)

        blocks_cpt = pack_texts(texts, tokenizer_cpt, args.seq_length)
        loader_cpt = make_block_loader(blocks_cpt, args.eval_batch_size, shuffle=False)
        loss_cpt = evaluate_loss(model_cpt, loader_cpt, args.device)

        results["base"][lang] = {"loss": loss_base, "perplexity": math.exp(loss_base)}
        results["cpt_best_weighted"][lang] = {"loss": loss_cpt, "perplexity": math.exp(loss_cpt)}
        print(
            f"[{lang}] base perplexity={results['base'][lang]['perplexity']:.2f} "
            f"| cpt perplexity={results['cpt_best_weighted'][lang]['perplexity']:.2f}"
        )

    for model_key in results:
        ppls = [results[model_key][lang]["perplexity"] for lang in LANGUAGES]
        results[model_key]["macro_average_perplexity"] = sum(ppls) / len(ppls)
        results[model_key]["worst_language_perplexity"] = max(ppls)

    macro_base = results["base"]["macro_average_perplexity"]
    macro_cpt = results["cpt_best_weighted"]["macro_average_perplexity"]
    results["relative_improvement_macro_perplexity"] = (macro_base - macro_cpt) / macro_base
    print(f"macro perplexity: base={macro_base:.2f} cpt={macro_cpt:.2f} (relative improvement {results['relative_improvement_macro_perplexity']:.2%})")

    del model_base, model_cpt
    if args.device == "cuda":
        torch.cuda.empty_cache()

    return results


def run_lm_eval(model_args, output_path, tasks, device):
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", ",".join(tasks),
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

    base_out = args.lm_eval_output_dir / "base"
    cpt_out = args.lm_eval_output_dir / "cpt_best_weighted"

    base_results = run_lm_eval(
        f"pretrained={args.model_name}", base_out, args.tasks, args.device
    )
    cpt_results = run_lm_eval(
        f"pretrained={args.model_name},peft={args.adapter_dir}", cpt_out, args.tasks, args.device
    )

    return {"base": base_results.get("results", {}), "cpt_best_weighted": cpt_results.get("results", {})}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=FINAL_MODEL_NAME)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
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
