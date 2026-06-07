"""SCIP solver wrapper for CVRP/TSP formulation.

Initializes SCIP models, adds variables/constraints, and handles the solve loop.
Supports adding custom branching rules.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from pyscipopt import Model, quicksum

from src.data.instance_generator import CVRPInstance


@dataclass
class SolveResult:
    """Statistics and results from a solver run."""
    cost: float
    nodes_explored: int
    solve_time: float
    optimality_gap: float
    status: str
    route: Optional[list[list[int]]] = None


class CVRPSolver:
    """Wraps PySCIPOpt to formulate and solve CVRP instances."""

    def __init__(self, instance: CVRPInstance, time_limit: float = 60.0, node_limit: int = 100_000):
        self.instance = instance
        self.model = Model(instance.instance_id)
        
        # Solver limits
        self.model.setParam("limits/time", time_limit)
        self.model.setParam("limits/nodes", node_limit)
        
        # Hide output by default
        self.model.hideOutput()
        
        self.vars = {}
        self._build_model()

    def _build_model(self):
        """Formulates the CVRP as a Mixed-Integer Program (MIP)."""
        n = self.instance.n_customers + 1
        dist = self.instance.distance_matrix
        
        # Variables: x_i_j = 1 if edge (i,j) is traversed
        for i in range(n):
            for j in range(n):
                if i != j:
                    self.vars[i, j] = self.model.addVar(
                        name=f"x_{i}_{j}", vtype="B", obj=dist[i, j]
                    )

        # Objective: minimize total distance (already set via obj param in addVar)
        self.model.setMinimize()

        # Constraints: visit each customer exactly once
        for i in range(1, n):
            # In-degree = 1
            self.model.addCons(
                quicksum(self.vars[j, i] for j in range(n) if i != j) == 1,
                name=f"visit_in_{i}"
            )
            # Out-degree = 1
            self.model.addCons(
                quicksum(self.vars[i, j] for j in range(n) if i != j) == 1,
                name=f"visit_out_{i}"
            )

        # Depot constraints
        # Sum of out-edges from depot = sum of in-edges to depot (flow balance handled above)
        # Note: CVRP might have multiple vehicles, but basic MTZ formulation
        # often uses a single flow or multiple loops.
        
        # Subtour elimination and Capacity (MTZ Formulation)
        # u_i: auxiliary variable for MTZ
        u = {}
        for i in range(1, n):
            u[i] = self.model.addVar(name=f"u_{i}", vtype="C", lb=self.instance.demands[i-1], ub=self.instance.capacity)

        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    self.model.addCons(
                        u[i] - u[j] + self.instance.capacity * self.vars[i, j] 
                        <= self.instance.capacity - self.instance.demands[j-1],
                        name=f"mtz_{i}_{j}"
                    )

    def add_branching_rule(self, rule):
        """Registers a custom branching rule callback."""
        # rule should be an instance of a class inheriting from pyscipopt.Branchrule
        self.model.includeBranchrule(
            rule, 
            rule.name, 
            rule.desc, 
            priority=rule.priority, 
            maxdepth=rule.maxdepth, 
            maxbounddist=rule.maxbounddist
        )

    def solve(self) -> SolveResult:
        """Runs the B&B search."""
        self.model.optimize()
        
        status = self.model.getStatus()
        cost = self.model.getObjVal() if status == "optimal" or status == "feasible" else float('inf')
        
        return SolveResult(
            cost=cost,
            nodes_explored=self.model.getNNodes(),
            solve_time=self.model.getSolvingTime(),
            optimality_gap=self.model.getGap(),
            status=status,
            route=None # Parsing logic could be added here
        )

    def get_lp_solution(self) -> dict[tuple[int, int], float]:
        """Returns fractional values of x_ij in current LP relaxation."""
        sol = {}
        for (i, j), var in self.vars.items():
            sol[i, j] = self.model.getVal(var)
        return sol
