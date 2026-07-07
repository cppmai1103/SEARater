"""
pipeline.md Section 12 - Final Continued Pretraining.

Trains Qwen2.5-1.5B-CPT-BestWeighted: LoRA-finetunes a fresh Qwen2.5-1.5B
Base on the final dataset built by build_final_cpt_dataset.py (Section 11,
the single best-weighted 50M-token/12.5M-per-language selection). The
pilot's other "final model" -- the untouched original Qwen2.5-1.5B Base --
needs no training at all; scripts/evaluate_models.py (Section 13) loads it
directly by name.

Usage:
    python3 scripts/run_final_cpt.py --device cuda
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sea_rater.cpt import (
    FINAL_MODEL_NAME,
    apply_lora,
    load_cpt_model_and_tokenizer,
    make_block_loader,
    pack_texts,
    train_one_epoch,
    trainable_parameters,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = REPO_ROOT / "data" / "candidate_corpus" / "final_cpt_dataset"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "models" / "qwen1.5b_cpt_bestweighted"

LANGUAGES = ["vi", "id", "th", "km"]


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
    parser.add_argument("--model-name", default=FINAL_MODEL_NAME)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--languages", nargs="+", default=LANGUAGES, choices=LANGUAGES)
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tokenizer, model = load_cpt_model_and_tokenizer(args.model_name, device=args.device)
    model = apply_lora(model, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
    model.print_trainable_parameters()

    texts = load_training_texts(args.dataset_dir, args.languages)
    print(f"packing {len(texts)} documents into {args.seq_length}-token blocks ...")
    blocks = pack_texts(texts, tokenizer, args.seq_length)
    print(f"{blocks.shape[0]} training blocks (~{blocks.numel() / 1e6:.2f}M tokens)")
    train_loader = make_block_loader(blocks, args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, args.device)
        print(f"epoch {epoch}/{args.epochs} | train loss {train_loss:.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter -> {args.output_dir}")


if __name__ == "__main__":
    main()
