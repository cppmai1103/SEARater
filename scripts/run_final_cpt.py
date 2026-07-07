"""
pipeline.md Section 12 - Final Continued Pretraining.

LoRA-finetunes a fresh Qwen2.5-1.5B Base on the final dataset built by
build_final_cpt_dataset.py (Section 11) for one baseline at a time --
default best_weighted (the pilot's main method), or one of Section 7's
random/clean_heuristic/equal_average baselines. The pilot's other
"final model" -- the untouched original Qwen2.5-1.5B Base -- needs no
training at all; scripts/evaluate_models.py (Section 13) loads it directly
by name.

Usage:
    python3 scripts/run_final_cpt.py --device cuda                          # best_weighted (default)
    python3 scripts/run_final_cpt.py --baseline random --device cuda
    python3 scripts/run_final_cpt.py --baseline clean_heuristic --device cuda
    python3 scripts/run_final_cpt.py --baseline equal_average --device cuda
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sea_rater.baselines import BASELINE_CHOICES, DATASET_DIR_BY_BASELINE, MODEL_DIR_BY_BASELINE
from sea_rater.cpt import (
    FINAL_MODEL_NAME,
    apply_lora,
    load_cpt_model_and_tokenizer,
    make_block_loader,
    pack_texts,
    train_one_epoch,
    trainable_parameters,
)
from sea_rater.languages import LANGUAGES


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_training_texts(dataset_dir, languages):
    texts = []
    for lang in languages:
        rows = load_jsonl(dataset_dir / f"{lang}.jsonl")
        texts.extend(row["text"] for row in rows)
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=BASELINE_CHOICES, default="best_weighted")
    parser.add_argument("--model-name", default=FINAL_MODEL_NAME)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--languages", nargs="+", default=LANGUAGES, choices=LANGUAGES)
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir or DATASET_DIR_BY_BASELINE[args.baseline]
    output_dir = args.output_dir or MODEL_DIR_BY_BASELINE[args.baseline]

    print(f"[{args.baseline}] Step 1/4: loading {args.model_name} + applying LoRA ...")
    tokenizer, model = load_cpt_model_and_tokenizer(args.model_name, device=args.device)
    model = apply_lora(model, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
    model.print_trainable_parameters()

    print(f"[{args.baseline}] Step 2/4: packing training data from {dataset_dir} ...")
    texts = load_training_texts(dataset_dir, args.languages)
    print(f"[{args.baseline}] packing {len(texts)} documents into {args.seq_length}-token blocks ...")
    blocks = pack_texts(texts, tokenizer, args.seq_length, desc=f"{args.baseline} pack")
    print(f"[{args.baseline}] {blocks.shape[0]} training blocks (~{blocks.numel() / 1e6:.2f}M tokens)")
    train_loader = make_block_loader(blocks, args.batch_size, shuffle=True)

    print(f"[{args.baseline}] Step 3/4: training for {args.epochs} epoch(s) ...")
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, args.device, desc=f"{args.baseline} epoch {epoch}/{args.epochs}"
        )
        print(f"[{args.baseline}] epoch {epoch}/{args.epochs} | train loss {train_loss:.4f}")

    print(f"[{args.baseline}] Step 4/4: saving LoRA adapter ...")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[{args.baseline}] Saved LoRA adapter -> {output_dir}")


if __name__ == "__main__":
    main()
