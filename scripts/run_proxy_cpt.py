"""
pipeline.md Section 9 - Proxy Continued Pretraining.

For each of the 16 weight combinations, continue-pretrains a fresh copy of
Qwen2.5-0.5B Base on that combination's selected data (from
select_weighted_corpus.py), starting from the identical checkpoint every
time and keeping every training setting fixed except the data -- then
evaluates each resulting model on the same fixed validation set (from
build_validation_set.py) and records per-language + macro + worst-language
loss to proxy_results.csv.

Usage:
    python3 scripts/run_proxy_cpt.py --device cuda
    python3 scripts/run_proxy_cpt.py --weight-ids W01 W06 --device cuda  # quick test
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sea_rater.cpt import (
    PROXY_MODEL_NAME,
    evaluate_loss,
    load_cpt_model_and_tokenizer,
    make_block_loader,
    pack_texts,
    train_one_epoch,
)
from sea_rater.dimensions import DIMENSIONS

REPO_ROOT = Path(__file__).resolve().parent.parent
SELECTED_DIR = REPO_ROOT / "data" / "candidate_corpus" / "selected"
VALIDATION_DIR = REPO_ROOT / "data" / "validation_set"
DEFAULT_WEIGHTS_PATH = REPO_ROOT / "data" / "weight_combinations.json"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "data" / "proxy_results.csv"

LANGUAGES = ["vi", "id", "th", "km"]
CSV_FIELDS = (
    ["weight_id"]
    + [f"w_{dim}" for dim in DIMENSIONS]
    + [f"loss_{lang}" for lang in LANGUAGES]
    + ["macro_loss", "worst_language_loss"]
)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_training_texts(weight_id):
    """All 4 languages' selected documents for one weight combination,
    combined into a single list -- language balance is already satisfied
    per-file by select_weighted_corpus.py's per-language token budget."""
    texts = []
    for lang in LANGUAGES:
        rows = load_jsonl(SELECTED_DIR / weight_id / f"{lang}.jsonl")
        texts.extend(row["text"] for row in rows)
    return texts


def build_validation_loaders(tokenizer, seq_length, eval_batch_size):
    """Pack the fixed validation set once per language; reused, unchanged,
    across all 16 weight combinations' evaluations."""
    loaders = {}
    for lang in LANGUAGES:
        rows = load_jsonl(VALIDATION_DIR / f"{lang}.jsonl")
        blocks = pack_texts([row["text"] for row in rows], tokenizer, seq_length)
        loaders[lang] = make_block_loader(blocks, eval_batch_size, shuffle=False)
    return loaders


def run_one_weight(combo, args, validation_loaders_factory):
    weight_id, weights = combo["id"], combo["weights"]
    print(f"\n=== {weight_id} ({combo['meaning']}) ===")

    tokenizer, model = load_cpt_model_and_tokenizer(args.model_name, device=args.device)

    texts = load_training_texts(weight_id)
    print(f"[{weight_id}] packing {len(texts)} documents into {args.seq_length}-token blocks ...")
    blocks = pack_texts(texts, tokenizer, args.seq_length)
    print(f"[{weight_id}] {blocks.shape[0]} training blocks (~{blocks.numel() / 1e6:.2f}M tokens)")
    train_loader = make_block_loader(blocks, args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_loss = train_one_epoch(model, train_loader, optimizer, args.device)
    print(f"[{weight_id}] train loss: {train_loss:.4f}")

    validation_loaders = validation_loaders_factory(tokenizer)
    losses = {}
    for lang, loader in validation_loaders.items():
        losses[lang] = evaluate_loss(model, loader, args.device)
        print(f"[{weight_id}] validation loss [{lang}]: {losses[lang]:.4f}")

    macro_loss = sum(losses.values()) / len(losses)
    worst_language_loss = max(losses.values())

    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()

    row = {"weight_id": weight_id}
    row.update({f"w_{dim}": weights[dim] for dim in DIMENSIONS})
    row.update({f"loss_{lang}": losses[lang] for lang in LANGUAGES})
    row["macro_loss"] = macro_loss
    row["worst_language_loss"] = worst_language_loss
    return row


def append_row(csv_path, row, write_header):
    mode = "w" if write_header else "a"
    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight-ids", nargs="+", default=None, help="subset of weight ids, e.g. W01 W06")
    parser.add_argument("--weights-path", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--model-name", default=PROXY_MODEL_NAME)
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    combos = json.loads(args.weights_path.read_text(encoding="utf-8"))
    if args.weight_ids:
        combos = [c for c in combos if c["id"] in args.weight_ids]
    print(f"Running proxy CPT for {len(combos)} weight combination(s)")

    # Validation blocks depend only on the tokenizer, which is the same
    # (Qwen2.5-0.5B's) for every run, so pack them once and reuse; still
    # exposed as a factory since each run loads its own tokenizer instance.
    validation_cache = {}

    def validation_loaders_factory(tokenizer):
        if not validation_cache:
            validation_cache["loaders"] = build_validation_loaders(
                tokenizer, args.seq_length, args.eval_batch_size
            )
        return validation_cache["loaders"]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    for i, combo in enumerate(combos):
        row = run_one_weight(combo, args, validation_loaders_factory)
        append_row(args.output_csv, row, write_header=(i == 0 and not args.output_csv.exists()))
        print(f"[{combo['id']}] macro_loss={row['macro_loss']:.4f} worst_language_loss={row['worst_language_loss']:.4f}")

    print(f"\nWrote proxy results -> {args.output_csv}")


if __name__ == "__main__":
    main()
