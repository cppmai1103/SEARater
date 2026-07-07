"""
pipeline.md Section 9 - Proxy Continued Pretraining.

For each of the 26 weight combinations, continue-pretrains a fresh copy of
Qwen2.5-0.5B Base on that combination's selected data (from
select_weighted_corpus.py), starting from the identical checkpoint every
time and keeping every training setting fixed except the data -- then
evaluates each resulting model on the same fixed validation set (from
build_validation_set.py) and records per-language + macro + worst-language
loss to proxy_results.csv.

Uses LoRA (same `apply_lora` helper as the final 1.5B run, `sea_rater/cpt.py`)
rather than full fine-tuning, so the proxy search trains under the same
regime the final CPT run will actually use -- otherwise a full-FT proxy
could rank weight combinations differently than a LoRA-constrained final
run would, since LoRA's low-rank updates don't necessarily respond to data
differences the same way full fine-tuning does.

Usage:
    python3 scripts/run_proxy_cpt.py --device cuda
    python3 scripts/run_proxy_cpt.py --weight-ids W01 W06 --device cuda  # quick test
"""

import argparse
import csv
import gc
import json
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sea_rater.cpt import (
    PROXY_MODEL_NAME,
    apply_lora,
    evaluate_loss,
    load_cpt_model_and_tokenizer,
    make_block_loader,
    pack_texts,
    train_one_epoch,
    trainable_parameters,
)
from sea_rater.dimensions import DIMENSIONS
from sea_rater.languages import LANGUAGES

REPO_ROOT = Path(__file__).resolve().parent.parent
SELECTED_DIR = REPO_ROOT / "data" / "candidate_corpus" / "selected"
VALIDATION_DIR = REPO_ROOT / "data" / "validation_set"
DEFAULT_WEIGHTS_PATH = REPO_ROOT / "data" / "weight_combinations.json"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "data" / "proxy_results.csv"

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
    """All languages' selected documents for one weight combination,
    combined into a single list -- language balance is already satisfied
    per-file by select_weighted_corpus.py's per-language token budget."""
    texts = []
    for lang in LANGUAGES:
        rows = load_jsonl(SELECTED_DIR / weight_id / f"{lang}.jsonl")
        texts.extend(row["text"] for row in rows)
    return texts


def build_validation_loaders(tokenizer, seq_length, eval_batch_size):
    """Pack the fixed validation set once per language; reused, unchanged,
    across all weight combinations' evaluations."""
    loaders = {}
    for lang in LANGUAGES:
        print(f"[validation] packing '{lang}' ...")
        rows = load_jsonl(VALIDATION_DIR / f"{lang}.jsonl")
        blocks = pack_texts(
            [row["text"] for row in rows], tokenizer, seq_length, desc=f"validation [{lang}] pack"
        )
        loaders[lang] = make_block_loader(blocks, eval_batch_size, shuffle=False)
    return loaders


def run_one_weight(combo, args, validation_loaders_factory):
    weight_id, weights = combo["id"], combo["weights"]
    print(f"\n=== {weight_id} ({combo['meaning']}) ===")

    print(f"[{weight_id}] loading {args.model_name} + applying LoRA ...")
    tokenizer, model = load_cpt_model_and_tokenizer(args.model_name, device=args.device)
    model = apply_lora(model, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
    model.print_trainable_parameters()

    texts = load_training_texts(weight_id)
    print(f"[{weight_id}] packing {len(texts)} documents into {args.seq_length}-token blocks ...")
    blocks = pack_texts(texts, tokenizer, args.seq_length, desc=f"{weight_id} pack")
    print(f"[{weight_id}] {blocks.shape[0]} training blocks (~{blocks.numel() / 1e6:.2f}M tokens)")
    train_loader = make_block_loader(blocks, args.batch_size, shuffle=True)

    print(f"[{weight_id}] training one epoch ...")
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=args.lr)
    train_loss = train_one_epoch(model, train_loader, optimizer, args.device, desc=f"{weight_id} train")
    print(f"[{weight_id}] train loss: {train_loss:.4f}")

    print(f"[{weight_id}] evaluating on validation set ...")
    validation_loaders = validation_loaders_factory(tokenizer)
    losses = {}
    for lang, loader in validation_loaders.items():
        losses[lang] = evaluate_loss(model, loader, args.device, desc=f"{weight_id} eval [{lang}]")
        print(f"[{weight_id}] validation loss [{lang}]: {losses[lang]:.4f}")

    macro_loss = sum(losses.values()) / len(losses)
    worst_language_loss = max(losses.values())

    # Drop every large reference before reclaiming: CUDA tensors from the
    # training/eval autograd graphs can form reference cycles that plain
    # refcounting won't clear, so gc.collect() has to run before
    # empty_cache() actually has anything to release. Without this, memory
    # creeps up across weight combinations until a later one OOMs even
    # though each individual run fits comfortably on its own.
    del model, optimizer, train_loader, blocks, validation_loaders
    if args.device == "cuda":
        gc.collect()
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
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
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
    write_header = not args.output_csv.exists()
    for combo in tqdm(combos, desc="weight combinations", unit="combo"):
        row = run_one_weight(combo, args, validation_loaders_factory)
        append_row(args.output_csv, row, write_header=write_header)
        write_header = False
        print(f"[{combo['id']}] macro_loss={row['macro_loss']:.4f} worst_language_loss={row['worst_language_loss']:.4f}")

    print(f"\nWrote proxy results -> {args.output_csv}")


if __name__ == "__main__":
    main()
