"""Quality rater heads: 5 independent MLP regressors over a frozen embedding."""

import torch
import torch.nn as nn

from sea_rater.dimensions import DIMENSIONS


def _make_head(input_dim, hidden_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )


class QualityRaterHeads(nn.Module):
    """5 separate MLP regression heads, one per PRRC+Cultural dimension.

    Kept as independent heads (rather than one shared output layer) so each
    dimension stays interpretable and can be weighted separately later
    (pipeline.md Section 8, the weighted-combination search).
    """

    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.heads = nn.ModuleDict(
            {dim: _make_head(input_dim, hidden_dim) for dim in DIMENSIONS}
        )

    def forward(self, embeddings):
        # Each head outputs (batch, 1); squeeze to (batch,).
        return {dim: self.heads[dim](embeddings).squeeze(-1) for dim in DIMENSIONS}


def huber_multi_loss(predictions, targets, delta=1.0):
    """Mean of per-dimension Huber losses (pipeline.md Section 4).

    `predictions` and `targets` are both dicts keyed by dimension name.
    The mean only gives the optimizer one scalar objective; the 5 heads
    remain independently trained regressors.
    """
    loss_fn = torch.nn.HuberLoss(delta=delta)
    losses = {dim: loss_fn(predictions[dim], targets[dim]) for dim in DIMENSIONS}
    total = torch.stack(list(losses.values())).mean()
    return total, losses
