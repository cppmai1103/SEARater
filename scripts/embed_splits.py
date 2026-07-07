"""
pipeline.md Section 4 - embedding step.

Embeds human_{train,dev,test}.jsonl (produced by split_human_annotations.py)
with the frozen multilingual-e5-large backbone and caches the result, so the
heads can be trained/re-trained many times without re-running the encoder.

Usage:
    python3 scripts/embed_splits.py --device cuda
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sea_rater.encoder import DEFAULT_MAX_LENGTH, DEFAULT_MODEL_NAME, embed_texts, load_encoder
from sea_rater.rater import DIMENSIONS

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "splits"
SPLITS = ["train", "dev", "test"]


def load_split(name):
    path = DATA_DIR / f"human_{name}.jsonl"
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = DATA_DIR / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_encoder(args.model_name, device=args.device)

    for split in SPLITS:
        rows = load_split(split)
        texts = [row["text"] for row in rows]
        embeddings = embed_texts(
            texts,
            tokenizer,
            model,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

        labels = {dim: torch.tensor([row[dim] for row in rows], dtype=torch.float32) for dim in DIMENSIONS}
        languages = [row["language"] for row in rows]
        doc_ids = [row["doc_id"] for row in rows]

        torch.save(
            {
                "embeddings": embeddings,
                "labels": labels,
                "languages": languages,
                "doc_ids": doc_ids,
                "model_name": args.model_name,
            },
            out_dir / f"{split}.pt",
        )
        print(f"{split}: embedded {len(rows)} docs -> {out_dir / f'{split}.pt'}")


if __name__ == "__main__":
    main()
