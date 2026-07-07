"""
pipeline.md Section 8 - Generate 16 Candidate Weight Combinations.

Materializes the pilot's fixed table of 16 interpretable weight vectors
over the 5 rater dimensions into data/weight_combinations.json. No random
sampling and no LightGBM here (pipeline.md is explicit: "No LightGBM is
used in the pilot. The best weight is chosen directly from proxy CPT
validation results.") -- this script just writes out the fixed table in a
form scripts/select_weighted_corpus.py (Section 7) and the proxy CPT step
(Section 9) can load.

Usage:
    python3 scripts/generate_weight_combinations.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sea_rater.dimensions import DIMENSIONS

DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "weight_combinations.json"

# (id, educational_value, reasoning, professionalism, cleanliness, cultural_nuance, meaning)
# Values match pipeline.md Section 8's table exactly.
WEIGHT_COMBINATIONS = [
    ("W01", 1.00, 0.00, 0.00, 0.00, 0.00, "Edu only"),
    ("W02", 0.00, 1.00, 0.00, 0.00, 0.00, "Reasoning only"),
    ("W03", 0.00, 0.00, 1.00, 0.00, 0.00, "Professionalism only"),
    ("W04", 0.00, 0.00, 0.00, 1.00, 0.00, "Cleanliness only"),
    ("W05", 0.00, 0.00, 0.00, 0.00, 1.00, "Cultural only"),
    ("W06", 0.20, 0.20, 0.20, 0.20, 0.20, "Equal average"),
    ("W07", 0.40, 0.20, 0.20, 0.10, 0.10, "Edu-heavy"),
    ("W08", 0.25, 0.35, 0.20, 0.10, 0.10, "Reasoning-heavy"),
    ("W09", 0.25, 0.15, 0.35, 0.15, 0.10, "Professionalism-heavy"),
    ("W10", 0.25, 0.10, 0.10, 0.40, 0.15, "Cleanliness-heavy"),
    ("W11", 0.25, 0.10, 0.10, 0.15, 0.40, "Cultural-heavy"),
    ("W12", 0.35, 0.30, 0.20, 0.10, 0.05, "Edu + reasoning"),
    ("W13", 0.35, 0.10, 0.30, 0.15, 0.10, "Edu + professionalism"),
    ("W14", 0.30, 0.15, 0.15, 0.30, 0.10, "Edu + cleanliness"),
    ("W15", 0.30, 0.15, 0.15, 0.10, 0.30, "Edu + cultural"),
    ("W16", 0.20, 0.30, 0.25, 0.15, 0.10, "Reasoning + professionalism"),
]


def build_combinations():
    combos = []
    for weight_id, edu, reasoning, prof, clean, cultural, meaning in WEIGHT_COMBINATIONS:
        weights = dict(
            zip(
                DIMENSIONS,
                [edu, reasoning, prof, clean, cultural],
            )
        )
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-9, f"{weight_id} weights sum to {total}, expected 1.0"
        assert all(w >= 0 for w in weights.values()), f"{weight_id} has a negative weight"
        combos.append({"id": weight_id, "meaning": meaning, "weights": weights})
    return combos


def main():
    combos = build_combinations()
    DEFAULT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT_PATH.write_text(json.dumps(combos, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(combos)} weight combinations -> {DEFAULT_OUTPUT_PATH}")
    for combo in combos:
        print(f"  {combo['id']:4s} {combo['weights']}  ({combo['meaning']})")


if __name__ == "__main__":
    main()
