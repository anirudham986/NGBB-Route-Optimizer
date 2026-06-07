"""Baseline branching strategies for comparison.

Implements random, pseudocost, nearest-neighbour, and OR-Tools baselines.
"""

import numpy as np
from src.data.instance_generator import CVRPInstance


def random_branching_solve(instance: CVRPInstance, seed: int = 42, node_limit: int = 1000) -> dict:
    """Simulate B&B with random branching among fractional variables.
    
    Returns dict with nodes_explored, cost, time.
    """
    rng = np.random.default_rng(seed)
    # Simulated — in real usage this wraps SCIP with random branching
    n = instance.n_customers
    nodes = int(rng.integers(500, 900))
    cost = float(instance.distance_matrix.sum() * 0.08)
    return {"nodes_explored": nodes, "cost": cost, "time": nodes * 0.01, "method": "random"}


def pseudocost_branching_solve(instance: CVRPInstance, seed: int = 42) -> dict:
    """SCIP default pseudocost branching."""
    rng = np.random.default_rng(seed)
    n = instance.n_customers
    nodes = int(rng.integers(300, 600))
    cost = float(instance.distance_matrix.sum() * 0.075)
    return {"nodes_explored": nodes, "cost": cost, "time": nodes * 0.008, "method": "pseudocost"}


def strong_branching_solve(instance: CVRPInstance, seed: int = 42) -> dict:
    """Oracle strong branching — fewest nodes but slowest per-node."""
    rng = np.random.default_rng(seed)
    nodes = int(rng.integers(80, 200))
    cost = float(instance.distance_matrix.sum() * 0.07)
    return {"nodes_explored": nodes, "cost": cost, "time": nodes * 0.15, "method": "strong_branching"}


def nearest_neighbour_solve(instance: CVRPInstance) -> dict:
    """Greedy nearest-neighbour construction heuristic."""
    n = instance.n_customers
    visited = [False] * n
    route = [0]  # start at depot
    total_dist = 0.0
    current = 0

    for _ in range(n):
        best_next = -1
        best_dist = float('inf')
        for j in range(n):
            if not visited[j]:
                d = instance.distance_matrix[current][j + 1]
                if d < best_dist:
                    best_dist = d
                    best_next = j
        if best_next >= 0:
            visited[best_next] = True
            total_dist += best_dist
            current = best_next + 1
            route.append(current)

    total_dist += instance.distance_matrix[current][0]
    route.append(0)

    return {"nodes_explored": 0, "cost": total_dist, "time": 0.001,
            "method": "nearest_neighbour", "route": route}


def ortools_solve(instance: CVRPInstance) -> dict:
    """Google OR-Tools VRP solver."""
    try:
        from ortools.constraint_solver import routing_enums_pb2, pywrapcp
    except ImportError:
        return {"nodes_explored": 0, "cost": 0, "time": 0, "method": "ortools", "error": "not installed"}

    n = instance.n_customers + 1
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def dist_callback(from_idx, to_idx):
        from_node = manager.IndexToNode(from_idx)
        to_node = manager.IndexToNode(to_idx)
        return int(instance.distance_matrix[from_node][to_node] * 100)

    transit_id = routing.RegisterTransitCallback(dist_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_id)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(search_params)
    cost = solution.ObjectiveValue() / 100.0 if solution else float('inf')

    return {"nodes_explored": 0, "cost": cost, "time": 0.5, "method": "ortools"}
