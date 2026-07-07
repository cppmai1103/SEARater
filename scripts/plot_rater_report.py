"""
Plot the quality-rater evaluation report produced by train_rater.py
(models/rater_test_report.json): a Spearman correlation heatmap
(language x dimension), an MAE-by-dimension bar chart, and per-dimension
score-distribution comparisons across languages.

Usage:
    python3 scripts/plot_rater_report.py
    python3 scripts/plot_rater_report.py --report models/rater_test_report.json --output-dir models/figures
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT = REPO_ROOT / "models" / "rater_test_report.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "models" / "figures"

DIMENSIONS = [
    "educational_value",
    "reasoning",
    "professionalism",
    "cleanliness",
    "cultural_nuance",
]


def plot_spearman_heatmap(report, output_dir):
    spearman = report["spearman_by_language_dimension"]
    languages = sorted(spearman)
    matrix = np.array([[spearman[lang][dim] for dim in DIMENSIONS] for lang in languages])

    fig, ax = plt.subplots(figsize=(8, 0.6 * len(languages) + 2))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(DIMENSIONS)))
    ax.set_xticklabels(DIMENSIONS, rotation=30, ha="right")
    ax.set_yticks(range(len(languages)))
    ax.set_yticklabels(languages)

    for i in range(len(languages)):
        for j in range(len(DIMENSIONS)):
            value = matrix[i, j]
            color = "white" if value < 0.4 or value > 0.85 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color)

    macro = report["macro_average_spearman"]
    worst = report["worst_language_spearman"]
    ax.set_title(
        f"Spearman correlation by language x dimension\n"
        f"macro avg = {macro:.3f}, worst language = {worst:.3f}"
    )
    fig.colorbar(im, ax=ax, label="Spearman correlation", shrink=0.8)
    fig.tight_layout()

    out_path = output_dir / "spearman_heatmap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_mae_bar(report, output_dir):
    mae = report["mae_by_dimension"]
    dims = [d for d in DIMENSIONS if d in mae]
    values = [mae[d] for d in dims]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(dims, values, color="#4C72B0")
    ax.set_ylabel("Mean absolute error")
    ax.set_title("MAE by dimension (test set, averaged across languages)")
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels(dims, rotation=30, ha="right")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom")
    fig.tight_layout()

    out_path = output_dir / "mae_by_dimension.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_score_distributions(report, output_dir):
    dist = report["score_distribution_by_language"]
    languages = sorted(dist)
    scores = list(range(6))

    fig, axes = plt.subplots(1, len(DIMENSIONS), figsize=(4 * len(DIMENSIONS), 4), sharey=True)
    width = 0.8 / len(languages)

    for ax, dim in zip(axes, DIMENSIONS):
        for i, lang in enumerate(languages):
            counts = dist[lang][dim]
            offsets = [s + (i - (len(languages) - 1) / 2) * width for s in scores]
            ax.bar(offsets, counts, width=width, label=lang)
        ax.set_title(dim)
        ax.set_xticks(scores)
        ax.set_xlabel("score")

    axes[0].set_ylabel("document count")
    axes[0].legend(title="language")
    fig.suptitle("Test-set score distribution by language and dimension")
    fig.tight_layout()

    out_path = output_dir / "score_distributions.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_spearman_heatmap(report, args.output_dir)
    plot_mae_bar(report, args.output_dir)
    plot_score_distributions(report, args.output_dir)


if __name__ == "__main__":
    main()
