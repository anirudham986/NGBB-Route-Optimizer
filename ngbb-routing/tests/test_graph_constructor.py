"""Tests for the bipartite graph constructor."""

import pytest
import torch
from src.data.instance_generator import InstanceGenerator
from src.data.graph_constructor import construct_bipartite_graph, VAR_FEATURE_DIM, CON_FEATURE_DIM


class TestGraphConstructor:
    def setup_method(self):
        gen = InstanceGenerator(seed=42)
        self.instance = gen.generate_random_euclidean(n_customers=10)

    def test_output_type(self):
        graph = construct_bipartite_graph(self.instance)
        assert hasattr(graph, "variable")
        assert hasattr(graph, "constraint")

    def test_variable_features_dim(self):
        graph = construct_bipartite_graph(self.instance)
        assert graph["variable"].x.shape[1] == VAR_FEATURE_DIM

    def test_constraint_features_dim(self):
        graph = construct_bipartite_graph(self.instance)
        assert graph["constraint"].x.shape[1] == CON_FEATURE_DIM

    def test_edge_indices_exist(self):
        graph = construct_bipartite_graph(self.instance)
        assert ("variable", "to", "constraint") in graph.edge_types
        assert ("constraint", "to", "variable") in graph.edge_types

    def test_n_variables_matches(self):
        graph = construct_bipartite_graph(self.instance)
        n = self.instance.n_customers + 1
        expected_vars = n * (n - 1)
        assert graph["variable"].x.shape[0] == expected_vars
