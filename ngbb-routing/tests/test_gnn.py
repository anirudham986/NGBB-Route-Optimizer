"""Tests for the GNN model."""

import pytest
import torch
from src.models.gnn import BipartiteGNN
from src.models.output_head import ScoringHead
from src.models.policy import NGBBPolicy
from src.data.instance_generator import InstanceGenerator
from src.data.graph_constructor import construct_bipartite_graph


class TestBipartiteGNN:
    def test_forward_shape(self):
        gen = InstanceGenerator(42)
        inst = gen.generate_random_euclidean(10)
        graph = construct_bipartite_graph(inst)
        gnn = BipartiteGNN(var_dim=12, con_dim=5, hidden_dim=64, n_layers=3)
        out = gnn(graph)
        assert out.shape == (graph["variable"].x.shape[0], 64)

    def test_scoring_head(self):
        head = ScoringHead(hidden_dim=64, dropout=0.0)
        x = torch.randn(100, 64)
        scores = head(x)
        assert scores.shape == (100,)

    def test_policy_forward(self):
        policy = NGBBPolicy.build()
        gen = InstanceGenerator(42)
        inst = gen.generate_random_euclidean(10)
        graph = construct_bipartite_graph(inst)
        logits = policy(graph)
        assert logits.shape[0] == graph["variable"].x.shape[0]

    def test_policy_score(self):
        policy = NGBBPolicy.build()
        gen = InstanceGenerator(42)
        inst = gen.generate_random_euclidean(10)
        graph = construct_bipartite_graph(inst)
        scores = policy.score(graph)
        assert scores.shape[0] == graph["variable"].x.shape[0]

    def test_policy_select(self):
        policy = NGBBPolicy.build()
        gen = InstanceGenerator(42)
        inst = gen.generate_random_euclidean(10)
        graph = construct_bipartite_graph(inst)
        idx = policy.select_branch_var(graph, [0, 1, 2, 3])
        assert idx in [0, 1, 2, 3]
