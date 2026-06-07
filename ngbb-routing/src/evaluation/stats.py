"""Statistical tests: Wilcoxon signed-rank, confidence intervals, result tables."""

import numpy as np
from scipy.stats import wilcoxon
import pandas as pd


def wilcoxon_test(ngbb_values: np.ndarray, baseline_values: np.ndarray) -> dict:
    """Wilcoxon signed-rank test for paired comparisons.
    
    Tests whether NGBB is significantly better (fewer nodes) than baseline.
    
    Returns:
        Dict with 'statistic', 'p_value', 'significant' (p < 0.05).
    """
    stat, p = wilcoxon(ngbb_values, baseline_values, alternative='less')
    return {"statistic": float(stat), "p_value": float(p), "significant": p < 0.05}


def confidence_interval(data: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """Compute confidence interval for the mean."""
    n = len(data)
    mean = np.mean(data)
    se = np.std(data, ddof=1) / np.sqrt(n)
    from scipy.stats import t
    h = se * t.ppf((1 + confidence) / 2, n - 1)
    return float(mean - h), float(mean + h)


def format_results_table(results: dict[str, dict]) -> str:
    """Format results as a markdown table.
    
    Args:
        results: Dict mapping method name -> metrics dict.
    
    Returns:
        Markdown table string.
    """
    rows = []
    for method, m in results.items():
        rows.append({
            "Method": method,
            "Nodes Explored": f"{m.get('nodes_mean', 0):.1f} ± {m.get('nodes_std', 0):.1f}",
            "Solve Time (s)": f"{m.get('time_mean', 0):.2f} ± {m.get('time_std', 0):.1f}",
            "Gap %": f"{m.get('gap_mean', 0):.2f}%",
            "Node Reduction": f"{m.get('reduction', 0):.1f}%",
        })
    
    df = pd.DataFrame(rows)
    return df.to_markdown(index=False)
