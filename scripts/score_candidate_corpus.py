"""
pipeline.md Section 5 - score the candidate corpus.

Loads data/candidate_corpus/{lang}.jsonl (built by build_candidate_corpus.py),
embeds each document with the frozen multilingual-e5-large encoder, and scores
it with the trained rater heads (models/rater_heads.pt) to produce
scored_corpus.jsonl per language with the 5 predicted quality dimensions.

This is GPU-bound (an encoder forward pass over up to 1M documents total) and
should be run on the cluster (see run_score_corpus.sh), not on the dev server.

Usage:
    python3 scripts/score_candidate_corpus.py --device cuda
    python3 scripts/score_candidate_corpus.py --languages vi --device cuda  # one language
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sea_rater.encoder import DEFAULT_MAX_LENGTH, DEFAULT_MODEL_NAME, embed_texts, load_encoder
from sea_rater.rater import DIMENSIONS, QualityRaterHeads

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "data" / "candidate_corpus"
DEFAULT_OUTPUT_DIR = CORPUS_DIR / "scored"
DEFAULT_RATER_CHECKPOINT = REPO_ROOT / "models" / "rater_heads.pt"

LANGUAGES = ["vi", "id", "th", "km"]


def load_rater(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = QualityRaterHeads(ckpt["input_dim"], hidden_dim=ckpt["hidden_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


def load_corpus(lang):
    path = CORPUS_DIR / f"{lang}.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@torch.no_grad()
def score_embeddings(model, embeddings, device, batch_size=256):
    scores = {dim: [] for dim in DIMENSIONS}
    for start in range(0, embeddings.shape[0], batch_size):
        batch = embeddings[start : start + batch_size].to(device)
        out = model(batch)
        for dim in DIMENSIONS:
            scores[dim].append(out[dim].cpu())
    return {dim: torch.cat(vals).tolist() for dim, vals in scores.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="+", default=LANGUAGES, choices=LANGUAGES)
    parser.add_argument("--rater-checkpoint", type=Path, default=DEFAULT_RATER_CHECKPOINT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--score-batch-size", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, encoder = load_encoder(args.model_name, device=args.device)
    rater = load_rater(args.rater_checkpoint, args.device)

    for lang in args.languages:
        rows = load_corpus(lang)
        texts = [row["text"] for row in rows]

        print(f"[{lang}] embedding {len(texts)} docs ...")
        embeddings = embed_texts(
            texts,
            tokenizer,
            encoder,
            device=args.device,
            batch_size=args.embed_batch_size,
            max_length=args.max_length,
        )

        print(f"[{lang}] scoring {len(texts)} docs ...")
        scores = score_embeddings(rater, embeddings, args.device, batch_size=args.score_batch_size)

        out_path = args.output_dir / f"{lang}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for i, row in enumerate(rows):
                record = {
                    "doc_id": row["doc_id"],
                    "language": row["language"],
                    "source": row.get("source"),
                    **{dim: scores[dim][i] for dim in DIMENSIONS},
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[{lang}] wrote {len(rows)} scored docs -> {out_path}\n")


if __name__ == "__main__":
    main()
