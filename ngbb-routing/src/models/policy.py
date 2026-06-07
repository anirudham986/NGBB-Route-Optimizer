"""Full NGBB policy: GNN + scoring head + inference utilities.

Combines the BipartiteGNN encoder with the ScoringHead to produce
a complete branching policy that can be used for both training and
SCIP integration.
"""

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from src.models.gnn import BipartiteGNN
from src.models.output_head import ScoringHead


class NGBBPolicy(nn.Module):
    """Neural-Guided Branch & Bound policy model."""

    def __init__(self, gnn: BipartiteGNN, head: ScoringHead):
        super().__init__()
        self.gnn = gnn
        self.head = head

    def forward(self, data: HeteroData) -> torch.Tensor:
        """Training forward pass: returns per-variable logits.

        Args:
            data: HeteroData bipartite graph.

        Returns:
            Logits shape (n_vars,) for cross-entropy loss.
        """
        embeddings = self.gnn(data)
        return self.head(embeddings)

    @torch.no_grad()
    def score(self, data: HeteroData) -> np.ndarray:
        """Inference: returns numpy scores array.

        Args:
            data: HeteroData bipartite graph.

        Returns:
            Numpy array of scores shape (n_vars,).
        """
        self.eval()
        logits = self.forward(data)
        return logits.cpu().numpy()

    @torch.no_grad()
    def select_branch_var(self, data: HeteroData, candidates: list[int]) -> int:
        """Select the highest-scoring candidate variable.

        Args:
            data: HeteroData bipartite graph.
            candidates: List of variable indices eligible for branching.

        Returns:
            Index of the best candidate variable.
        """
        scores = self.score(data)
        candidate_scores = [(c, scores[c]) for c in candidates if c < len(scores)]
        if not candidate_scores:
            return candidates[0]
        return max(candidate_scores, key=lambda x: x[1])[0]

    @staticmethod
    def build(var_dim=12, con_dim=5, hidden_dim=64, n_layers=3, dropout=0.1):
        """Factory method to create a policy with default architecture."""
        gnn = BipartiteGNN(var_dim, con_dim, hidden_dim, n_layers)
        head = ScoringHead(hidden_dim, dropout)
        return NGBBPolicy(gnn, head)
