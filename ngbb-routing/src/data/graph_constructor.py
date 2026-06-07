"""Converts CVRPInstance + LP state into bipartite graph for GNN input.

Graph structure:
  - Variable nodes (one per decision variable x_ij): 12D features
  - Constraint nodes (one per constraint): 5D features
"""

import numpy as np
import torch
from torch_geometric.data import HeteroData
from src.data.instance_generator import CVRPInstance

VAR_FEATURE_DIM = 12
CON_FEATURE_DIM = 5


def construct_bipartite_graph(instance, lp_solution=None, pseudocosts=None, constraint_info=None):
    """Convert CVRP instance + LP state to PyG HeteroData bipartite graph."""
    n = instance.n_customers + 1
    dist = instance.distance_matrix
    max_dist = max(dist.max(), 1.0)

    # Build edge list (all directed i->j, i!=j)
    edges = [(i, j) for i in range(n) for j in range(n) if i != j]
    n_vars = len(edges)
    costs = np.array([dist[i][j] for i, j in edges])
    cost_range = max(costs.max() - costs.min(), 1.0)
    max_demand = max(instance.demands.max(), 1.0)

    # Variable features (12D)
    vf = np.zeros((n_vars, VAR_FEATURE_DIM), dtype=np.float32)
    for idx, (i, j) in enumerate(edges):
        lp = lp_solution.get((i, j), 0.0) if lp_solution else 0.5
        vf[idx, 0] = lp
        vf[idx, 1] = min(lp, 1.0 - lp)
        if pseudocosts and (i, j) in pseudocosts:
            vf[idx, 2], vf[idx, 3], vf[idx, 4] = pseudocosts[(i, j)]
        vf[idx, 5] = (dist[i][j] - costs.min()) / cost_range
        vf[idx, 6] = dist[i][j] / max_dist
        vf[idx, 7] = n - 1
        vf[idx, 8] = n - 1
        vf[idx, 9] = 1.0 if (i == 0 or j == 0) else 0.0
        vf[idx, 10] = (0.0 if i == 0 else instance.demands[i - 1]) / max_demand
        vf[idx, 11] = (0.0 if j == 0 else instance.demands[j - 1]) / max_demand

    # Constraint nodes
    if constraint_info is None:
        constraint_info = _default_constraints(instance, n)
    n_cons = len(constraint_info)
    cf = np.zeros((n_cons, CON_FEATURE_DIM), dtype=np.float32)
    for idx, c in enumerate(constraint_info):
        cf[idx] = [c.get("rhs", 1.0), c.get("activity", 0.5),
                   c.get("slack", 0.5), c.get("type", 0), 1.0]

    # Bipartite edges
    v2c_s, v2c_d, c2v_s, c2v_d = [], [], [], []
    for ci, c in enumerate(constraint_info):
        for vi in c.get("involved_edges", range(min(n_vars, 20))):
            if vi < n_vars:
                v2c_s.append(vi); v2c_d.append(ci)
                c2v_s.append(ci); c2v_d.append(vi)

    data = HeteroData()
    data["variable"].x = torch.tensor(vf)
    data["constraint"].x = torch.tensor(cf)
    data["variable", "to", "constraint"].edge_index = torch.tensor([v2c_s, v2c_d], dtype=torch.long)
    data["constraint", "to", "variable"].edge_index = torch.tensor([c2v_s, c2v_d], dtype=torch.long)
    data.edge_list = edges
    data.n_variables = n_vars
    data.n_constraints = n_cons
    return data


def _default_constraints(instance, n):
    """Generate default constraints from a CVRP instance."""
    constraints = []
    n_v = max(2, int(np.ceil(instance.demands.sum() / instance.capacity)))
    ne = n * (n - 1)
    # Capacity constraints
    for v in range(n_v):
        constraints.append({"rhs": float(instance.capacity), "activity": float(instance.demands.sum() / n_v),
                            "slack": float(instance.capacity - instance.demands.sum() / n_v), "type": 0,
                            "involved_edges": list(range(v * (n - 1), min((v + 1) * (n - 1), ne)))})
    # Covering constraints
    for c in range(instance.n_customers):
        cn = c + 1
        inv = [ei for ei, (i, j) in enumerate((i, j) for i in range(n) for j in range(n) if i != j) if i == cn or j == cn]
        constraints.append({"rhs": 2.0, "activity": 1.0, "slack": 1.0, "type": 1, "involved_edges": inv})
    # Subtour constraints
    for s in range(min(instance.n_customers, 10)):
        constraints.append({"rhs": float(s + 1), "activity": float((s + 2) * 0.4),
                            "slack": float(s + 1 - (s + 2) * 0.4), "type": 2,
                            "involved_edges": list(range(s * 5, min(s * 5 + 10, ne)))})
    return constraints
