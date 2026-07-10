"""
Analysis: summarizes data/evaluation_results.json (scripts/evaluate_models.py's
output) into one markdown table per task that's actually complete -- 13.1
held-out perplexity plus whichever of sib200/belebele/mmlu have at least a
"base" result -- comparing base against every trained baseline
(best_weighted/random/clean_heuristic/equal_average). Each downstream task
gets two tables, one per metric evaluate_models.py computes:
`accuracy` (argmax correctness, binary per doc) and
`token_normalized_prob_correct` (continuous confidence in specifically the
gold answer -- moves even when accuracy's 0/1 looks flat between two
checkpoints, per docs/09_evaluation.md's rationale for tracking both).

Tasks with no data yet (e.g. belebele before its first successful run) are
listed as not-yet-evaluated rather than omitted, so the summary always
reflects the full intended scope of Section 13, not just what happens to be
done today. Re-run any time evaluation_results.json changes.

Usage:
    python3 analysis/final_eval_summary.py
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO_ROOT / "data" / "evaluation_results.json"
OUTPUT_PATH = Path(__file__).resolve().parent / "final_eval_summary.md"

BASELINE_ORDER = ["best_weighted", "random", "clean_heuristic", "equal_average"]
ALL_DOWNSTREAM_TASKS = ["sib200", "belebele", "mmlu"]

# metric name -> (macro key, relative-improvement key) in evaluate_models.py's
# rollup_downstream_task/add_relative_improvement output. The two metrics'
# relative-improvement keys aren't named symmetrically (accuracy's has no
# "accuracy" in it, for historical reasons), so this is the source of truth
# rather than deriving one name from the other.
METRICS = {
    "accuracy": {"macro_key": "macro_average_accuracy", "rel_key": "relative_improvement_vs_base",
                 "column": "macro accuracy"},
    "token_normalized_prob_correct": {
        "macro_key": "macro_average_token_normalized_prob_correct",
        "rel_key": "token_normalized_prob_correct_relative_improvement_vs_base",
        "column": "macro token_normalized_prob_correct",
    },
}


def fmt_pct(x):
    return f"{x:+.2%}" if x is not None else "—"


def perplexity_rows(results):
    base = results.get("base", {}).get("held_out_lm_eval")
    if not base:
        return []
    rows = [("base", base["macro_average_perplexity"], base["worst_language_perplexity"], None)]
    for baseline in BASELINE_ORDER:
        cpt = results.get(baseline, {}).get("held_out_lm_eval")
        if cpt:
            rows.append((baseline, cpt["macro_average_perplexity"], cpt["worst_language_perplexity"],
                         cpt.get("relative_improvement_vs_base")))
    return rows


def downstream_task_rows(results, task_name, metric):
    """Returns (rows, bucket_label) for the given metric ("accuracy" or
    "token_normalized_prob_correct"); rows is empty if not even `base` has
    this task yet."""
    spec = METRICS[metric]
    base_task = results.get("base", {}).get("downstream_eval", {}).get(task_name)
    rows, worst_key = [], None
    if base_task:
        worst_key = next(k for k in base_task if k.startswith("worst_") and k.endswith(f"_{metric}"))
        rows.append(("base", base_task[spec["macro_key"]], base_task[worst_key], None))
    for baseline in BASELINE_ORDER:
        cpt_task = results.get(baseline, {}).get("downstream_eval", {}).get(task_name)
        if not cpt_task:
            continue
        if worst_key is None:
            worst_key = next(k for k in cpt_task if k.startswith("worst_") and k.endswith(f"_{metric}"))
        rows.append((baseline, cpt_task[spec["macro_key"]], cpt_task[worst_key], cpt_task.get(spec["rel_key"])))
    bucket_label = "subject" if worst_key and "subject" in worst_key else "language"
    return rows, bucket_label


def render_markdown(results):
    lines = [
        "# Final-eval summary (auto-generated)",
        "",
        "Source: `data/evaluation_results.json`. "
        "Regenerate with `python3 analysis/final_eval_summary.py`.",
        "",
        "## 13.1 — held-out perplexity",
        "",
    ]

    ppl_rows = perplexity_rows(results)
    if not ppl_rows:
        lines += ["_Not yet evaluated._", ""]
    else:
        lines += [
            "| model | macro perplexity | worst-language | rel. improvement vs base |",
            "|---|---:|---:|---:|",
        ]
        for model, macro, worst, rel in ppl_rows:
            lines.append(f"| {model} | {macro:.2f} | {worst:.2f} | {fmt_pct(rel)} |")
        lines.append("")

    for task_name in ALL_DOWNSTREAM_TASKS:
        lines += [f"## {task_name}", ""]
        any_rows = False
        for metric, spec in METRICS.items():
            rows, bucket_label = downstream_task_rows(results, task_name, metric)
            if not rows:
                continue
            any_rows = True
            lines += [f"### {metric}", ""]
            lines += [
                f"| model | {spec['column']} | worst {bucket_label} | rel. improvement vs base |",
                "|---|---:|---:|---:|",
            ]
            for model, macro, worst, rel in rows:
                lines.append(f"| {model} | {macro:.4f} | {worst:.4f} | {fmt_pct(rel)} |")
            done = {r[0] for r in rows}
            missing = [b for b in BASELINE_ORDER if b not in done]
            if missing:
                lines += ["", f"_Still missing for `{task_name}`/{metric}: {', '.join(missing)}._"]
            lines.append("")
        if not any_rows:
            lines += ["_Not yet evaluated for any model._", ""]

    return "\n".join(lines)


def main():
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    markdown = render_markdown(results)
    OUTPUT_PATH.write_text(markdown + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
