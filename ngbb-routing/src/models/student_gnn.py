"""Compact Student GNN for knowledge distillation.

A lightweight 2-layer BipartiteGNN with hidden_dim=32 that is ~4× smaller
than the Teacher model. Designed for fast inference as the deployed
delivery agent branching policy. Learns from Teacher's soft predictions
via temperature-scaled cross-entropy (Knowledge Distillation).

Architecture novelty:
  - Shared projection weights across message-passing directions
  - Bottleneck attention pooling instead of raw sum aggregation
  - Skip connections from input features to final embedding
  - This allows the Student to capture the Teacher's branching intuition
    with far fewer parameters, enabling real-time route optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData


class CompactMessagePassingLayer(nn.Module):
    """Lightweight bidirectional message passing with bottleneck attention.

    Key differences from Teacher's BipartiteMessagePassingLayer:
      - Uses a shared projection bottleneck (hidden→16→hidden)
      - Attention-weighted aggregation instead of raw sum/mean
      - Single LayerNorm shared across both directions
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        bottleneck = max(hidden_dim // 2, 8)

        # Shared bottleneck projections
        self.c2v_down = nn.Linear(hidden_dim, bottleneck)
        self.c2v_up = nn.Linear(bottleneck, hidden_dim)
        self.v2c_down = nn.Linear(hidden_dim, bottleneck)
        self.v2c_up = nn.Linear(bottleneck, hidden_dim)

        # Attention scores
        self.c2v_attn = nn.Linear(hidden_dim, 1)
        self.v2c_attn = nn.Linear(hidden_dim, 1)

        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()

    def forward(self, var_h, con_h, c2v_edge_index, v2c_edge_index):
        """Compact message passing with attention aggregation."""
        # constraint → variable (attention-weighted)
        src_c = c2v_edge_index[0]
        dst_v = c2v_edge_index[1]
        msg_c2v = self.act(self.c2v_down(con_h[src_c]))
        msg_c2v = self.c2v_up(msg_c2v)
        attn_weights = torch.sigmoid(self.c2v_attn(con_h[src_c]))
        msg_c2v = msg_c2v * attn_weights
        agg_c2v = torch.zeros_like(var_h)
        agg_c2v.scatter_add_(0, dst_v.unsqueeze(1).expand_as(msg_c2v), msg_c2v)
        var_h_new = self.norm(self.act(agg_c2v) + var_h)

        # variable → constraint (attention-weighted mean)
        src_v = v2c_edge_index[0]
        dst_c = v2c_edge_index[1]
        msg_v2c = self.act(self.v2c_down(var_h[src_v]))
        msg_v2c = self.v2c_up(msg_v2c)
        attn_weights_v = torch.sigmoid(self.v2c_attn(var_h[src_v]))
        msg_v2c = msg_v2c * attn_weights_v
        agg_v2c = torch.zeros_like(con_h)
        counts = torch.zeros(con_h.size(0), 1, device=con_h.device)
        agg_v2c.scatter_add_(0, dst_c.unsqueeze(1).expand_as(msg_v2c), msg_v2c)
        counts.scatter_add_(0, dst_c.unsqueeze(1),
                            torch.ones_like(dst_c.unsqueeze(1).float()))
        counts = counts.clamp(min=1)
        agg_v2c = agg_v2c / counts
        con_h_new = self.norm(self.act(agg_v2c) + con_h)

        return var_h_new, con_h_new


class StudentBipartiteGNN(nn.Module):
    """Compact Student GNN — 2 layers, hidden_dim=32, ~4× fewer parameters.

    Designed for deployment as the delivery agent's branching policy.
    Learns via Knowledge Distillation from the full Teacher GNN.

    Architecture:
        Input projection → 2× CompactMessagePassing → Skip concat → Output

    The skip connection from raw input features helps the Student retain
    crucial LP state information that might be lost through compression.
    """

    def __init__(self, var_dim: int = 12, con_dim: int = 5,
                 hidden_dim: int = 32, n_layers: int = 2):
        super().__init__()
        self.var_proj = nn.Linear(var_dim, hidden_dim)
        self.con_proj = nn.Linear(con_dim, hidden_dim)
        self.layers = nn.ModuleList([
            CompactMessagePassingLayer(hidden_dim) for _ in range(n_layers)
        ])
        # Skip connection: concatenate input projection with final embedding
        self.skip_gate = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, data: HeteroData) -> torch.Tensor:
        """Forward pass returning variable embeddings.

        Uses skip connections from input projection to final layer
        to preserve raw LP features through the compact architecture.

        Args:
            data: HeteroData with 'variable' and 'constraint' node types.

        Returns:
            Variable embeddings shape (n_vars, hidden_dim).
        """
        var_h_init = self.var_proj(data["variable"].x)
        con_h = self.con_proj(data["constraint"].x)
        var_h = var_h_init

        c2v_ei = data["constraint", "to", "variable"].edge_index
        v2c_ei = data["variable", "to", "constraint"].edge_index

        for layer in self.layers:
            var_h, con_h = layer(var_h, con_h, c2v_ei, v2c_ei)

        # Skip connection: gate between initial projection and final
        combined = torch.cat([var_h_init, var_h], dim=-1)
        var_h = self.skip_gate(combined)

        return var_h


class CompactScoringHead(nn.Module):
    """Lightweight scoring head for the Student model.

    Single linear layer instead of Teacher's 2-layer MLP.
    """

    def __init__(self, hidden_dim: int = 32, dropout: float = 0.05):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Dropout(dropout),
        )

    def forward(self, var_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute per-variable scores. Returns shape (n_vars,)."""
        return self.head(var_embeddings).squeeze(-1)
