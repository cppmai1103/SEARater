"""
pipeline.md Section 10 - Select Final Weight Combination.

Reads proxy_results.csv (run_proxy_cpt.py's output) and picks the weight
combination with the lowest macro validation loss. If multiple
combinations land within --tie-tolerance of the best macro loss, the one
with the better (lower) worst-language loss wins the tie. No LightGBM /
regression here -- pipeline.md is explicit that the pilot picks directly
from the 16 proxy CPT results.

Usage:
    python3 scripts/select_best_weight.py
"""

import argparse
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_CSV = REPO_ROOT / "data" / "proxy_results.csv"
DEFAULT_WEIGHTS_PATH = REPO_ROOT / "data" / "weight_combinations.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "best_weight.json"

DEFAULT_TIE_TOLERANCE = 0.01  # relative gap in macro_loss counted as "very close"


def load_results(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for row in rows:
        for key in row:
            if key != "weight_id":
                row[key] = float(row[key])
    return rows


def select_best(rows, tie_tolerance):
    best_macro_loss = min(row["macro_loss"] for row in rows)
    threshold = best_macro_loss * (1 + tie_tolerance)
    contenders = [row for row in rows if row["macro_loss"] <= threshold]
    return min(contenders, key=lambda row: row["worst_language_loss"]), contenders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--weights-path", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--tie-tolerance", type=float, default=DEFAULT_TIE_TOLERANCE)
    args = parser.parse_args()

    rows = load_results(args.results_csv)
    print(f"Loaded {len(rows)} proxy result(s) from {args.results_csv}")

    best, contenders = select_best(rows, args.tie_tolerance)
    if len(contenders) > 1:
        print(
            f"{len(contenders)} weight combination(s) within {args.tie_tolerance:.1%} of the best "
            f"macro_loss ({min(r['macro_loss'] for r in rows):.4f}); "
            f"broke the tie by worst_language_loss."
        )

    combos = json.loads(args.weights_path.read_text(encoding="utf-8"))
    meaning = next((c["meaning"] for c in combos if c["id"] == best["weight_id"]), None)
    weights = next((c["weights"] for c in combos if c["id"] == best["weight_id"]), None)

    result = {
        "weight_id": best["weight_id"],
        "meaning": meaning,
        "weights": weights,
        "macro_loss": best["macro_loss"],
        "worst_language_loss": best["worst_language_loss"],
        "per_language_loss": {
            key[len("loss_") :]: value for key, value in best.items() if key.startswith("loss_")
        },
    }

    args.output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Best weight: {best['weight_id']} ({meaning})")
    print(f"  macro_loss={best['macro_loss']:.4f}  worst_language_loss={best['worst_language_loss']:.4f}")
    print(f"Wrote {args.output_path}")


if __name__ == "__main__":
    main()
