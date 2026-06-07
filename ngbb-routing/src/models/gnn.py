"""Bipartite GNN for branching variable scoring.

Message-passing architecture operating on the constraint-variable
bipartite graph. Alternates messages between node types with
residual connections and layer normalization.
"""

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData


class BipartiteMessagePassingLayer(nn.Module):
    """Single round of bidirectional message passing on bipartite graph."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        # constraint -> variable
        self.c2v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.c2v_norm = nn.LayerNorm(hidden_dim)
        # variable -> constraint
        self.v2c_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v2c_norm = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, var_h, con_h, c2v_edge_index, v2c_edge_index):
        """One round of message passing.

        Args:
            var_h: Variable embeddings (n_vars, hidden_dim)
            con_h: Constraint embeddings (n_cons, hidden_dim)
            c2v_edge_index: (2, E1) constraint->variable edges
            v2c_edge_index: (2, E2) variable->constraint edges

        Returns:
            Updated (var_h, con_h)
        """
        # constraint -> variable: sum aggregation
        src_c = c2v_edge_index[0]  # constraint indices
        dst_v = c2v_edge_index[1]  # variable indices
        msg_c2v = self.c2v_linear(con_h[src_c])
        agg_c2v = torch.zeros_like(var_h)
        agg_c2v.scatter_add_(0, dst_v.unsqueeze(1).expand_as(msg_c2v), msg_c2v)
        var_h_new = self.c2v_norm(self.relu(agg_c2v) + var_h)  # residual

        # variable -> constraint: mean aggregation
        src_v = v2c_edge_index[0]  # variable indices
        dst_c = v2c_edge_index[1]  # constraint indices
        msg_v2c = self.v2c_linear(var_h[src_v])
        agg_v2c = torch.zeros_like(con_h)
        counts = torch.zeros(con_h.size(0), 1, device=con_h.device)
        agg_v2c.scatter_add_(0, dst_c.unsqueeze(1).expand_as(msg_v2c), msg_v2c)
        counts.scatter_add_(0, dst_c.unsqueeze(1), torch.ones_like(dst_c.unsqueeze(1).float()))
        counts = counts.clamp(min=1)
        agg_v2c = agg_v2c / counts
        con_h_new = self.v2c_norm(self.relu(agg_v2c) + con_h)  # residual

        return var_h_new, con_h_new


class BipartiteGNN(nn.Module):
    """Bipartite GNN operating on constraint-variable graphs.

    Architecture:
        Input projection -> N rounds of BipartiteMessagePassing -> output embeddings
    """

    def __init__(self, var_dim: int = 12, con_dim: int = 5,
                 hidden_dim: int = 64, n_layers: int = 3):
        super().__init__()
        self.var_proj = nn.Linear(var_dim, hidden_dim)
        self.con_proj = nn.Linear(con_dim, hidden_dim)
        self.layers = nn.ModuleList([
            BipartiteMessagePassingLayer(hidden_dim) for _ in range(n_layers)
        ])

    def forward(self, data: HeteroData) -> torch.Tensor:
        """Forward pass returning variable node embeddings.

        Args:
            data: HeteroData with 'variable' and 'constraint' node types.

        Returns:
            Variable embeddings shape (n_vars, hidden_dim).
        """
        var_h = self.var_proj(data["variable"].x)
        con_h = self.con_proj(data["constraint"].x)

        c2v_ei = data["constraint", "to", "variable"].edge_index
        v2c_ei = data["variable", "to", "constraint"].edge_index

        for layer in self.layers:
            var_h, con_h = layer(var_h, con_h, c2v_ei, v2c_ei)

        return var_h
