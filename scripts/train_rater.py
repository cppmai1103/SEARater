"""
pipeline.md Section 4 - train the human-only quality rater heads.

Trains 5 independent MLP regression heads on top of cached, frozen
multilingual-e5-large embeddings (see embed_splits.py), using Huber loss
per head and AdamW updating only the heads.

Usage:
    python3 scripts/train_rater.py --epochs 50 --lr 1e-3
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sea_rater.rater import DIMENSIONS, QualityRaterHeads, huber_multi_loss

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "splits" / "embeddings"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def load_embeddings(split):
    return torch.load(DATA_DIR / f"{split}.pt", weights_only=False)


def make_loader(bundle, batch_size, shuffle):
    tensors = [bundle["embeddings"]] + [bundle["labels"][dim] for dim in DIMENSIONS]
    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def unpack_batch(batch, device):
    embeddings = batch[0].to(device)
    targets = {dim: batch[i + 1].to(device) for i, dim in enumerate(DIMENSIONS)}
    return embeddings, targets


@torch.no_grad()
def predict_all(model, bundle, device, batch_size=256):
    model.eval()
    loader = make_loader(bundle, batch_size, shuffle=False)
    preds = defaultdict(list)
    for batch in loader:
        embeddings, _ = unpack_batch(batch, device)
        out = model(embeddings)
        for dim in DIMENSIONS:
            preds[dim].append(out[dim].cpu())
    return {dim: torch.cat(vals).numpy() for dim, vals in preds.items()}


def evaluate(model, bundle, device):
    """Spearman/MAE per language & dimension, macro-average, worst-language (Section 4)."""
    preds = predict_all(model, bundle, device)
    languages = np.array(bundle["languages"])
    unique_langs = sorted(set(languages))

    spearman_table = {}  # lang -> dim -> rho
    mae_table = defaultdict(list)  # dim -> [per-lang mae]
    for lang in unique_langs:
        mask = languages == lang
        spearman_table[lang] = {}
        for dim in DIMENSIONS:
            y_true = bundle["labels"][dim].numpy()[mask]
            y_pred = preds[dim][mask]
            rho = spearmanr(y_true, y_pred).correlation
            spearman_table[lang][dim] = float(rho) if not np.isnan(rho) else 0.0
            mae_table[dim].append(float(np.mean(np.abs(y_true - y_pred))))

    all_rhos = [spearman_table[lang][dim] for lang in unique_langs for dim in DIMENSIONS]
    macro_spearman = float(np.mean(all_rhos))
    worst_lang_spearman = min(
        float(np.mean(list(spearman_table[lang].values()))) for lang in unique_langs
    )
    mae_per_dim = {dim: float(np.mean(vals)) for dim, vals in mae_table.items()}

    score_distribution = {
        lang: {
            dim: np.bincount(
                bundle["labels"][dim].numpy()[languages == lang].astype(int), minlength=6
            ).tolist()
            for dim in DIMENSIONS
        }
        for lang in unique_langs
    }

    return {
        "spearman_by_language_dimension": spearman_table,
        "macro_average_spearman": macro_spearman,
        "worst_language_spearman": worst_lang_spearman,
        "mae_by_dimension": mae_per_dim,
        "score_distribution_by_language": score_distribution,
    }


def train(args):
    device = args.device
    print("loading cached embeddings for train/dev/test ...")
    train_bundle = load_embeddings("train")
    dev_bundle = load_embeddings("dev")
    test_bundle = load_embeddings("test")

    input_dim = train_bundle["embeddings"].shape[1]
    model = QualityRaterHeads(input_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_loader = make_loader(train_bundle, args.batch_size, shuffle=True)

    best_macro_spearman = -1.0
    best_state = None
    epochs_without_improvement = 0

    print(f"training for up to {args.epochs} epoch(s) (patience {args.patience}) ...")
    epoch_progress = tqdm(range(1, args.epochs + 1), desc="epochs", unit="epoch")
    for epoch in epoch_progress:
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch} train", unit="batch", leave=False):
            embeddings, targets = unpack_batch(batch, device)
            optimizer.zero_grad()
            preds = model(embeddings)
            loss, _ = huber_multi_loss(preds, targets, delta=args.huber_delta)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * embeddings.size(0)
        epoch_loss /= len(train_loader.dataset)

        dev_metrics = evaluate(model, dev_bundle, device)
        macro_spearman = dev_metrics["macro_average_spearman"]
        epoch_progress.set_postfix(train_loss=f"{epoch_loss:.4f}", dev_macro_spearman=f"{macro_spearman:.4f}")
        tqdm.write(
            f"epoch {epoch:3d} | train_loss {epoch_loss:.4f} "
            f"| dev_macro_spearman {macro_spearman:.4f}"
        )

        if macro_spearman > best_macro_spearman:
            best_macro_spearman = macro_spearman
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                tqdm.write(f"early stopping at epoch {epoch} (best dev macro spearman {best_macro_spearman:.4f})")
                break

    model.load_state_dict(best_state)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": best_state, "input_dim": input_dim, "hidden_dim": args.hidden_dim},
        MODELS_DIR / "rater_heads.pt",
    )

    print("evaluating on test set ...")
    test_metrics = evaluate(model, test_bundle, device)
    report_path = MODELS_DIR / "rater_test_report.json"
    report_path.write_text(json.dumps(test_metrics, indent=2, ensure_ascii=False))

    print("\n=== Test set evaluation ===")
    print(f"macro_average_spearman: {test_metrics['macro_average_spearman']:.4f}")
    print(f"worst_language_spearman: {test_metrics['worst_language_spearman']:.4f}")
    print(f"mae_by_dimension: {test_metrics['mae_by_dimension']}")
    print(f"Full report written to {report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train(args)


if __name__ == "__main__":
    main()
