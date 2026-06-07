"""Evaluation metrics: node reduction, solve time ratio, optimality gap."""

import numpy as np


def node_reduction(nodes_baseline: int, nodes_ngbb: int) -> float:
    """Percentage reduction in explored nodes.
    
    Formula: (nodes_baseline - nodes_ngbb) / nodes_baseline * 100
    Target: >= 60%
    """
    if nodes_baseline == 0:
        return 0.0
    return (nodes_baseline - nodes_ngbb) / nodes_baseline * 100


def solve_time_ratio(time_ngbb: float, time_baseline: float) -> float:
    """Ratio of NGBB solve time to baseline. Target: < 2.0x."""
    if time_baseline == 0:
        return float('inf')
    return time_ngbb / time_baseline


def optimality_gap(cost: float, optimal: float) -> float:
    """Percentage gap from optimal. Target: <= 0.5%."""
    if optimal == 0:
        return 0.0
    return abs(cost - optimal) / abs(optimal) * 100


def generalization_ratio(reduction_large: float, reduction_id: float) -> float:
    """Ratio of OOD to ID node reduction. Target: >= 0.7."""
    if reduction_id == 0:
        return 0.0
    return reduction_large / reduction_id


def compute_all_metrics(results: list[dict]) -> dict:
    """Compute aggregate metrics from a list of solve results.
    
    Args:
        results: List of dicts with keys: nodes_baseline, nodes_ngbb,
                 time_baseline, time_ngbb, cost_ngbb, optimal_cost.
    
    Returns:
        Dict with mean ± std for each metric.
    """
    nr = [node_reduction(r["nodes_baseline"], r["nodes_ngbb"]) for r in results]
    st = [solve_time_ratio(r["time_ngbb"], r["time_baseline"]) for r in results]
    og = [optimality_gap(r["cost_ngbb"], r["optimal_cost"]) for r in results]

    return {
        "node_reduction_mean": np.mean(nr),
        "node_reduction_std": np.std(nr),
        "solve_time_ratio_mean": np.mean(st),
        "solve_time_ratio_std": np.std(st),
        "optimality_gap_mean": np.mean(og),
        "optimality_gap_std": np.std(og),
        "n_instances": len(results),
    }
