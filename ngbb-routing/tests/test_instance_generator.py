"""Tests for the CVRP instance generator."""

import numpy as np
import pytest
from src.data.instance_generator import InstanceGenerator, CVRPInstance


class TestInstanceGenerator:
    def test_random_euclidean_shape(self):
        gen = InstanceGenerator(seed=42)
        inst = gen.generate_random_euclidean(n_customers=20)
        assert inst.n_customers == 20
        assert inst.depot.shape == (2,)
        assert inst.customers.shape == (20, 2)
        assert inst.demands.shape == (20,)
        assert inst.distance_matrix.shape == (21, 21)

    def test_random_euclidean_depot_position(self):
        gen = InstanceGenerator(seed=42)
        inst = gen.generate_random_euclidean(n_customers=10)
        np.testing.assert_array_equal(inst.depot, [50.0, 50.0])

    def test_random_euclidean_deterministic(self):
        g1 = InstanceGenerator(seed=99)
        g2 = InstanceGenerator(seed=99)
        i1 = g1.generate_random_euclidean(15)
        i2 = g2.generate_random_euclidean(15)
        np.testing.assert_array_equal(i1.customers, i2.customers)
        np.testing.assert_array_equal(i1.demands, i2.demands)

    def test_clustered_shape(self):
        gen = InstanceGenerator(seed=42)
        inst = gen.generate_clustered(n_customers=30, n_clusters=4)
        assert inst.n_customers == 30
        assert inst.customers.shape == (30, 2)
        assert np.all(inst.customers >= 0) and np.all(inst.customers <= 100)

    def test_mixed_shape(self):
        gen = InstanceGenerator(seed=42)
        inst = gen.generate_mixed(n_customers=25, cluster_ratio=0.6)
        assert inst.n_customers == 25

    def test_demands_range(self):
        gen = InstanceGenerator(seed=42)
        inst = gen.generate_random_euclidean(n_customers=50)
        assert np.all(inst.demands >= 1)
        assert np.all(inst.demands <= 10)

    def test_distance_matrix_symmetric(self):
        gen = InstanceGenerator(seed=42)
        inst = gen.generate_random_euclidean(n_customers=10)
        np.testing.assert_allclose(inst.distance_matrix, inst.distance_matrix.T, atol=1e-10)

    def test_distance_matrix_zero_diagonal(self):
        gen = InstanceGenerator(seed=42)
        inst = gen.generate_random_euclidean(n_customers=10)
        np.testing.assert_array_equal(np.diag(inst.distance_matrix), 0.0)

    def test_instance_id_format(self):
        gen = InstanceGenerator(seed=42)
        inst = gen.generate_random_euclidean(n_customers=20)
        assert inst.instance_id == "random_20_42"
        inst2 = gen.generate_clustered(n_customers=15)
        assert inst2.instance_id == "clustered_15_42"
