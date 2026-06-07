"""Serializes a B&B node state into a PyG HeteroData graph.

Provides the bridge between live SCIP model state and the
graph constructor used by the GNN policy.
"""

import torch
from torch_geometric.data import HeteroData

from src.data.feature_extractor import FeatureExtractor
from src.data.graph_constructor import construct_bipartite_graph
from src.data.instance_generator import CVRPInstance


class StateSerializer:
    """Converts live SCIP model state to a PyG graph for GNN inference."""

    def __init__(self, instance: CVRPInstance):
        self.instance = instance
        self.extractor = FeatureExtractor()

    def serialize(self, model) -> HeteroData:
        """Extract current B&B node state and return as HeteroData.

        Args:
            model: Active PySCIPOpt Model during solve.

        Returns:
            HeteroData bipartite graph ready for GNN forward pass.
        """
        lp_sol = self.extractor.extract_lp_solution(model)
        pseudocosts = self.extractor.extract_pseudocosts(model)
        constraint_info = self.extractor.extract_constraints(model)

        graph = construct_bipartite_graph(
            self.instance,
            lp_solution=lp_sol,
            pseudocosts=pseudocosts,
            constraint_info=constraint_info if constraint_info else None,
        )

        return graph

    def serialize_with_label(self, model, label_var_idx: int) -> dict:
        """Serialize state and attach a branching label for training.

        Args:
            model: Active PySCIPOpt Model.
            label_var_idx: Index of the variable chosen by oracle.

        Returns:
            Dict with 'graph' (HeteroData) and 'label' (int).
        """
        graph = self.serialize(model)
        return {"graph": graph, "label": label_var_idx}

    @staticmethod
    def save_sample(sample: dict, filepath: str) -> None:
        """Save a serialized sample to disk as .pt file."""
        torch.save(sample, filepath)

    @staticmethod
    def load_sample(filepath: str) -> dict:
        """Load a serialized sample from disk."""
        return torch.load(filepath)
