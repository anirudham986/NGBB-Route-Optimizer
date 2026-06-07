"""CLI: Full evaluation suite.

Usage:
    python src/evaluation/benchmark.py --checkpoint checkpoints/best.pt --test-splits test_small test_large
"""

from pathlib import Path

import click
import json
import numpy as np

from src.evaluation.metrics import compute_all_metrics, node_reduction
from src.evaluation.stats import wilcoxon_test, format_results_table
from src.utils.logging import get_logger

logger = get_logger("ngbb.evaluation.benchmark")


@click.command()
@click.option("--checkpoint", required=True, help="Path to model checkpoint")
@click.option("--test-splits", multiple=True, default=["test_small", "test_large"])
@click.option("--baselines", multiple=True, default=["random", "pseudocost", "strong_branching"])
@click.option("--output-dir", default="results/", help="Output directory")
@click.option("--seeds", default=5, type=int, help="Number of random seeds")
def main(checkpoint, test_splits, baselines, output_dir, seeds):
    """Run full evaluation benchmark."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info(f"Evaluating {checkpoint} on splits: {test_splits}")
    logger.info(f"Baselines: {baselines}, Seeds: {seeds}")

    # Placeholder: In full implementation, load model, load test data,
    # run each baseline and NGBB on each instance, collect metrics.
    
    all_results = {}
    for split in test_splits:
        logger.info(f"--- Evaluating on {split} ---")
        
        # Example results structure (would be filled by actual evaluation)
        split_results = {
            "split": split,
            "methods": {},
            "statistical_tests": {},
        }
        
        for baseline in baselines:
            logger.info(f"Running baseline: {baseline}")
            # Would run actual baseline here
            
        logger.info("Running NGBB")
        # Would run NGBB here
        
        all_results[split] = split_results

    # Save results
    results_file = out / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
