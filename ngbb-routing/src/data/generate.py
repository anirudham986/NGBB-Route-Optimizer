"""CLI for generating and saving CVRP instance datasets.

Generates training, validation, and test splits of synthetic instances
using the InstanceGenerator, with multiprocessing support.

Usage:
    python src/data/generate.py --split train --n-instances 50000 --size-min 20 --size-max 50
    python src/data/generate.py --split val_ood --n-instances 2000 --size-min 50 --size-max 100
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
import numpy as np
from tqdm import tqdm

from src.data.instance_generator import InstanceGenerator, CVRPInstance
from src.utils.logging import get_logger

logger = get_logger("ngbb.data.generate")


def _serialize_instance(instance: CVRPInstance) -> dict:
    """Convert a CVRPInstance to a JSON-serializable dictionary."""
    return {
        "instance_id": instance.instance_id,
        "n_customers": instance.n_customers,
        "depot": instance.depot.tolist(),
        "customers": instance.customers.tolist(),
        "demands": instance.demands.tolist(),
        "capacity": instance.capacity,
        "distance_matrix": instance.distance_matrix.tolist(),
    }


def _generate_single(args: tuple) -> dict:
    """Generate a single instance (for multiprocessing).

    Args:
        args: Tuple of (seed, n_customers, capacity, gen_type).

    Returns:
        Serialized instance dictionary.
    """
    seed, n_customers, capacity, gen_type = args
    gen = InstanceGenerator(seed=seed)

    if gen_type == "random":
        instance = gen.generate_random_euclidean(n_customers, capacity)
    elif gen_type == "clustered":
        instance = gen.generate_clustered(n_customers, capacity=capacity)
    elif gen_type == "mixed":
        instance = gen.generate_mixed(n_customers, capacity=capacity)
    else:
        raise ValueError(f"Unknown generation type: {gen_type}")

    return _serialize_instance(instance)


@click.command()
@click.option("--split", required=True, type=str,
              help="Dataset split name (train, val_id, val_ood, test_small, test_large)")
@click.option("--n-instances", required=True, type=int,
              help="Number of instances to generate")
@click.option("--size-min", default=20, type=int,
              help="Minimum number of customer nodes")
@click.option("--size-max", default=50, type=int,
              help="Maximum number of customer nodes")
@click.option("--capacity", default=50, type=int,
              help="Vehicle capacity Q")
@click.option("--workers", default=4, type=int,
              help="Number of parallel worker processes")
@click.option("--output-dir", default="data/generated/", type=str,
              help="Output directory for generated instances")
@click.option("--seed-offset", default=0, type=int,
              help="Seed offset for this generation batch")
def main(
    split: str,
    n_instances: int,
    size_min: int,
    size_max: int,
    capacity: int,
    workers: int,
    output_dir: str,
    seed_offset: int,
):
    """Generate CVRP instances for the specified dataset split."""
    output_path = Path(output_dir) / split
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Generating {n_instances} instances for split '{split}' "
        f"(n={size_min}-{size_max}, Q={capacity}, workers={workers})"
    )

    # Prepare generation tasks
    rng = np.random.default_rng(42 + seed_offset)
    gen_types = ["random", "clustered", "mixed"]

    tasks = []
    for i in range(n_instances):
        seed = int(rng.integers(0, 2**31))
        n_customers = int(rng.integers(size_min, size_max, endpoint=True))
        gen_type = gen_types[i % len(gen_types)]
        tasks.append((seed, n_customers, capacity, gen_type))

    # Generate with progress bar
    instances = []
    if workers <= 1:
        for task in tqdm(tasks, desc=f"Generating {split}"):
            instances.append(_generate_single(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_generate_single, t): i for i, t in enumerate(tasks)}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Generating {split}"):
                instances.append(future.result())

    # Save as chunked JSON files (10K instances per file)
    chunk_size = 10_000
    for chunk_idx in range(0, len(instances), chunk_size):
        chunk = instances[chunk_idx:chunk_idx + chunk_size]
        chunk_file = output_path / f"instances_{chunk_idx:06d}.json"
        with open(chunk_file, "w", encoding="utf-8") as f:
            json.dump(chunk, f)
        logger.info(f"Saved {len(chunk)} instances to {chunk_file}")

    # Save metadata
    meta = {
        "split": split,
        "n_instances": len(instances),
        "size_range": [size_min, size_max],
        "capacity": capacity,
        "seed_offset": seed_offset,
    }
    meta_file = output_path / "metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Generation complete: {len(instances)} instances in {output_path}")


if __name__ == "__main__":
    main()
