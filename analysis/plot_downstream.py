"""
Analysis: DELTA plots (each baseline's score minus base's score, for both
metrics evaluate_models.py computes -- `accuracy` and
`token_normalized_prob_correct`) from data/evaluation_results.json. `base`
itself is dropped from every chart (it's the implicit zero line every delta
is measured against) so the plot reads as "how much did this baseline move,
and in which direction," which is the actual question -- the aggregate
macro/worst numbers in final_eval_summary.md don't show *where* a baseline
wins or loses.

sib200/belebele are broken out per-language (e.g. which languages drive
equal_average's -27% sib200 macro regression). mmlu is treated as one
aggregate set, not broken out per-subject (57 subjects isn't a useful plot
axis for a general-ability forgetting check) -- one macro-delta bar per
baseline, same as any other downstream task's headline number.

Requires matplotlib/numpy, which aren't on the bare `python3` on this dev
server -- run with the sea_rater_env interpreter:

Usage:
    /Utilisateurs/pchau/.conda/envs/sea_rater_env/bin/python3 analysis/plot_downstream.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO_ROOT / "data" / "evaluation_results.json"
OUTPUT_DIR = Path(__file__).resolve().parent

# Fixed baseline order/hues -- dataviz skill's validated categorical
# palette (light mode), slots assigned by model identity and never
# re-ordered by rank. `base` isn't in here: it's the delta's zero line, not
# a plotted series.
BASELINE_ORDER = ["best_weighted", "random", "clean_heuristic", "equal_average"]
MODEL_COLOR = {
    "best_weighted": "#2a78d6",  # categorical slot 1, blue
    "random": "#1baf7a",  # slot 2, aqua
    "clean_heuristic": "#eda100",  # slot 3, yellow
    "equal_average": "#008300",  # slot 4, green
}
BUCKET_LABEL_BY_TASK = {"sib200": "language", "belebele": "language", "mmlu": "subject"}
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def load_results():
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(MUTED)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=SECONDARY_INK, labelsize=9)


def per_bucket_deltas(results, task_name, metric):
    """{baseline: {bucket: baseline_value - base_value}} for every baseline
    that has this task, plus the sorted bucket list -- or (None, None) if
    `base` itself doesn't have this task/metric yet (nothing to delta
    against)."""
    per_key = f"per_{BUCKET_LABEL_BY_TASK[task_name]}_{metric}"
    base_task = results.get("base", {}).get("downstream_eval", {}).get(task_name)
    if not base_task or per_key not in base_task:
        return None, None
    base_per_bucket = base_task[per_key]
    buckets = sorted(base_per_bucket.keys())

    deltas = {}
    for baseline in BASELINE_ORDER:
        cpt_task = results.get(baseline, {}).get("downstream_eval", {}).get(task_name)
        if not cpt_task or per_key not in cpt_task:
            continue
        cpt_per_bucket = cpt_task[per_key]
        deltas[baseline] = {b: cpt_per_bucket[b] - base_per_bucket[b] for b in buckets}
    return deltas, buckets


def plot_language_delta(results, task_name, metric, title, out_name):
    deltas, buckets = per_bucket_deltas(results, task_name, metric)
    if not deltas:
        print(f"[{task_name}/{metric}] no data yet, skipping plot")
        return

    baselines_present = [m for m in BASELINE_ORDER if m in deltas]
    n_models = len(baselines_present)
    x = np.arange(len(buckets))
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    ax.axhline(0, color=MUTED, linewidth=1.2, zorder=2)

    for i, model in enumerate(baselines_present):
        values = [deltas[model][b] for b in buckets]
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, values, width=width * 0.92, label=model, color=MODEL_COLOR[model], zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(buckets, color=INK)
    ax.set_ylabel(f"Δ {metric} vs base", color=SECONDARY_INK)
    ax.set_title(title, color=INK, fontsize=12, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=n_models)
    fig.tight_layout()

    out_path = OUTPUT_DIR / out_name
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_mmlu_delta_aggregate(results, metric):
    """mmlu treated as one aggregate set (not broken out per-subject) --
    one bar per baseline, height = macro_average_<metric> delta vs base,
    same headline-number treatment as sib200/belebele's summary rows."""
    task_name = "mmlu"
    macro_key = f"macro_average_{metric}"
    base_task = results.get("base", {}).get("downstream_eval", {}).get(task_name)
    if not base_task or macro_key not in base_task:
        print(f"[mmlu/{metric}] no data yet, skipping plot")
        return
    base_value = base_task[macro_key]

    baselines_present, deltas = [], []
    for baseline in BASELINE_ORDER:
        cpt_task = results.get(baseline, {}).get("downstream_eval", {}).get(task_name)
        if not cpt_task or macro_key not in cpt_task:
            continue
        baselines_present.append(baseline)
        deltas.append(cpt_task[macro_key] - base_value)
    if not baselines_present:
        print(f"[mmlu/{metric}] no baseline data yet, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    ax.axhline(0, color=MUTED, linewidth=1.2, zorder=2)

    x = np.arange(len(baselines_present))
    ax.bar(x, deltas, width=0.6, color=[MODEL_COLOR[m] for m in baselines_present], zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(baselines_present, color=INK, rotation=15, ha="right")
    ax.set_ylabel(f"Δ macro {metric} vs base", color=SECONDARY_INK)
    ax.set_title(f"mmlu — Δ macro {metric} vs base (forgetting check)",
                 color=INK, fontsize=12, fontweight="bold", loc="left")
    fig.tight_layout()

    out_path = OUTPUT_DIR / f"mmlu_delta_{metric}.png"
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    results = load_results()
    for metric in ["accuracy", "token_normalized_prob_correct"]:
        plot_language_delta(results, "sib200", metric, f"sib200 — Δ {metric} vs base, by language",
                             f"sib200_delta_{metric}.png")
        plot_language_delta(results, "belebele", metric, f"belebele — Δ {metric} vs base, by language",
                             f"belebele_delta_{metric}.png")
        plot_mmlu_delta_aggregate(results, metric)


if __name__ == "__main__":
    main()
