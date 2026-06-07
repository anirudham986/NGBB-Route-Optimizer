"""Tests for evaluation metrics."""

import numpy as np
import pytest
from src.evaluation.metrics import node_reduction, solve_time_ratio, optimality_gap, generalization_ratio


class TestMetrics:
    def test_node_reduction_basic(self):
        assert node_reduction(800, 200) == 75.0

    def test_node_reduction_zero_baseline(self):
        assert node_reduction(0, 100) == 0.0

    def test_node_reduction_equal(self):
        assert node_reduction(500, 500) == 0.0

    def test_solve_time_ratio(self):
        assert solve_time_ratio(2.0, 4.0) == 0.5

    def test_optimality_gap_zero(self):
        assert optimality_gap(100.0, 100.0) == 0.0

    def test_optimality_gap_nonzero(self):
        assert abs(optimality_gap(101.0, 100.0) - 1.0) < 1e-6

    def test_generalization_ratio(self):
        assert generalization_ratio(50.0, 70.0) == pytest.approx(50.0 / 70.0)
