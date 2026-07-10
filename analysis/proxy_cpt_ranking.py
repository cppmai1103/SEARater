"""
Analysis: ranks all 26 proxy-CPT weight combinations (pipeline.md Section 9)
by held-out macro validation loss. Reuses scripts/select_best_weight.py's
own load_results/select_best functions rather than re-implementing the
tie-break rule, so this can never disagree with what actually got written
to data/best_weight.json.

Re-run any time data/proxy_results.csv changes.

Usage:
    python3 analysis/proxy_cpt_ranking.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from select_best_weight import DEFAULT_TIE_TOLERANCE, load_results, select_best  # noqa: E402

RESULTS_CSV = REPO_ROOT / "data" / "proxy_results.csv"
WEIGHTS_PATH = REPO_ROOT / "data" / "weight_combinations.json"
BEST_WEIGHT_PATH = REPO_ROOT / "data" / "best_weight.json"
OUTPUT_PATH = Path(__file__).resolve().parent / "proxy_cpt_ranking.md"


def meanings_by_id(weights_path):
    combos = json.loads(weights_path.read_text(encoding="utf-8"))
    return {c["id"]: c["meaning"] for c in combos}


def render_markdown(rows, contenders, best, meanings, recorded_best):
    contender_ids = {r["weight_id"] for r in contenders}
    ranked = sorted(rows, key=lambda r: r["macro_loss"])

    lines = [
        "# Proxy CPT ranking (auto-generated)",
        "",
        f"Source: `data/proxy_results.csv` ({len(rows)} weight combinations). "
        f"Regenerate with `python3 analysis/proxy_cpt_ranking.py`.",
        "",
        f"**Selected** (`data/best_weight.json`): `{best['weight_id']}` "
        f"({meanings.get(best['weight_id'], '?')}) — "
        f"macro_loss={best['macro_loss']:.4f}, worst_language_loss={best['worst_language_loss']:.4f}"
        + (
            f", breaking a {len(contenders)}-way tie (all within "
            f"{DEFAULT_TIE_TOLERANCE:.0%} of the lowest macro_loss) by worst_language_loss."
            if len(contenders) > 1
            else "."
        ),
        "",
    ]

    if recorded_best.get("weight_id") and recorded_best["weight_id"] != best["weight_id"]:
        lines += [
            f"**Note:** `data/best_weight.json` currently records `{recorded_best['weight_id']}`, which "
            f"disagrees with recomputing from the current `proxy_results.csv` (`{best['weight_id']}`) — "
            f"re-run `scripts/select_best_weight.py` to refresh it.",
            "",
        ]

    lines += [
        "| rank | weight_id | meaning | macro_loss | worst_language_loss | tie group? | selected |",
        "|---:|---|---|---:|---:|:---:|:---:|",
    ]
    for i, row in enumerate(ranked, start=1):
        selected = "**Y**" if row["weight_id"] == best["weight_id"] else ""
        tie = "Y" if row["weight_id"] in contender_ids else ""
        lines.append(
            f"| {i} | {row['weight_id']} | {meanings.get(row['weight_id'], '?')} | "
            f"{row['macro_loss']:.4f} | {row['worst_language_loss']:.4f} | {tie} | {selected} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    rows = load_results(RESULTS_CSV)
    best, contenders = select_best(rows, DEFAULT_TIE_TOLERANCE)
    meanings = meanings_by_id(WEIGHTS_PATH)
    recorded_best = json.loads(BEST_WEIGHT_PATH.read_text(encoding="utf-8")) if BEST_WEIGHT_PATH.exists() else {}

    markdown = render_markdown(rows, contenders, best, meanings, recorded_best)
    OUTPUT_PATH.write_text(markdown + "\n", encoding="utf-8")
    print(f"Loaded {len(rows)} proxy result(s); selected {best['weight_id']} ({meanings.get(best['weight_id'])})")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
