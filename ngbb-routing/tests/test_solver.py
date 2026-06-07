"""Tests for the SCIP solver wrapper (requires PySCIPOpt)."""

import pytest

try:
    from pyscipopt import Model
    HAS_SCIP = True
except ImportError:
    HAS_SCIP = False

from src.data.instance_generator import InstanceGenerator


@pytest.mark.skipif(not HAS_SCIP, reason="PySCIPOpt not installed")
class TestSolver:
    def test_scip_available(self):
        m = Model()
        assert m is not None

    def test_solver_creates_model(self):
        from src.solver.scip_wrapper import CVRPSolver
        gen = InstanceGenerator(42)
        inst = gen.generate_random_euclidean(5)
        solver = CVRPSolver(inst, time_limit=10.0, node_limit=100)
        assert solver.model is not None

    def test_solver_solves_small(self):
        from src.solver.scip_wrapper import CVRPSolver
        gen = InstanceGenerator(42)
        inst = gen.generate_random_euclidean(5)
        solver = CVRPSolver(inst, time_limit=30.0, node_limit=1000)
        result = solver.solve()
        assert result.status in ["optimal", "feasible", "infeasible", "timelimit", "nodelimit"]
