"""MLP scoring head for per-variable branching scores.

Takes variable embeddings from the GNN and produces a scalar
score for each variable. No softmax — raw logits for cross-entropy.
"""

import torch
import torch.nn as nn


class ScoringHead(nn.Module):
    """MLP head: hidden_dim -> 32 -> 1 per variable node."""

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Dropout(dropout),
        )

    def forward(self, var_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute per-variable scores.

        Args:
            var_embeddings: (n_vars, hidden_dim) from GNN.

        Returns:
            Scores shape (n_vars,). Raw logits, no softmax.
        """
        return self.mlp(var_embeddings).squeeze(-1)
