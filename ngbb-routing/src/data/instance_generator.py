"""Core CVRP/TSP instance generator.

Generates synthetic instances for training, validation, and testing.
Supports random Euclidean, clustered, mixed, and TSPLIB format parsing.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.utils.prng import SeededRNG


@dataclass
class CVRPInstance:
    """A single Capacitated Vehicle Routing Problem instance.

    Attributes:
        n_customers: Number of customer nodes (excludes depot).
        depot: Depot coordinates, shape (2,).
        customers: Customer coordinates, shape (n_customers, 2).
        demands: Customer demands, shape (n_customers,).
        capacity: Vehicle capacity Q.
        distance_matrix: Pairwise Euclidean distances, shape (n_customers+1, n_customers+1).
                        Index 0 = depot, indices 1..n = customers.
        instance_id: Unique identifier string "{type}_{n}_{seed}".
    """

    n_customers: int
    depot: np.ndarray
    customers: np.ndarray
    demands: np.ndarray
    capacity: int
    distance_matrix: np.ndarray
    instance_id: str


def _compute_distance_matrix(depot: np.ndarray, customers: np.ndarray) -> np.ndarray:
    """Compute full Euclidean distance matrix including depot.

    Args:
        depot: Depot coords shape (2,).
        customers: Customer coords shape (n, 2).

    Returns:
        Distance matrix shape (n+1, n+1). Index 0 = depot.
    """
    all_nodes = np.vstack([depot.reshape(1, 2), customers])
    n = all_nodes.shape[0]
    diff = all_nodes[:, np.newaxis, :] - all_nodes[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    return dist


class InstanceGenerator:
    """Generates CVRP/TSP problem instances deterministically from a seed.

    All methods are deterministic given the same seed. Supports:
    - Random Euclidean placement
    - Gaussian-clustered placement
    - Mixed (clustered + random)
    - TSPLIB .vrp file parsing

    Usage:
        gen = InstanceGenerator(seed=42)
        instance = gen.generate_random_euclidean(n_customers=20)
    """

    def __init__(self, seed: int):
        """Initialize the generator with a seed.

        Args:
            seed: Integer seed for full reproducibility.
        """
        self.seed = seed
        self._rng = SeededRNG(seed)

    def generate_random_euclidean(
        self, n_customers: int, capacity: int = 50
    ) -> CVRPInstance:
        """Generate an instance with uniform random node positions in [0, 100]^2.

        Depot is fixed at (50, 50). Customer demands are sampled
        from Uniform(1, 10).

        Args:
            n_customers: Number of customer nodes.
            capacity: Vehicle capacity Q.

        Returns:
            CVRPInstance with random Euclidean layout.
        """
        depot = np.array([50.0, 50.0])
        customers = self._rng.uniform(0.0, 100.0, size=(n_customers, 2))
        demands = self._rng.integers(1, 10, size=n_customers)
        dist_matrix = _compute_distance_matrix(depot, customers)

        return CVRPInstance(
            n_customers=n_customers,
            depot=depot,
            customers=customers,
            demands=demands,
            capacity=capacity,
            distance_matrix=dist_matrix,
            instance_id=f"random_{n_customers}_{self.seed}",
        )

    def generate_clustered(
        self, n_customers: int, n_clusters: int = 3, capacity: int = 50
    ) -> CVRPInstance:
        """Generate an instance with Gaussian-clustered node positions.

        Each cluster has a random centre in [10, 90]^2 with spread ~ N(0, 8).
        Depot is fixed at (50, 50).

        Args:
            n_customers: Number of customer nodes.
            n_clusters: Number of Gaussian clusters.
            capacity: Vehicle capacity Q.

        Returns:
            CVRPInstance with clustered layout.
        """
        depot = np.array([50.0, 50.0])

        # Generate cluster centres
        centres = self._rng.uniform(10.0, 90.0, size=(n_clusters, 2))

        # Assign customers to clusters roughly evenly
        assignments = np.arange(n_customers) % n_clusters

        # Generate customer positions around their cluster centres
        customers = np.zeros((n_customers, 2))
        for i in range(n_customers):
            cluster_id = assignments[i]
            customers[i] = self._rng.normal(
                loc=centres[cluster_id], scale=8.0, size=2
            )

        # Clip to valid range [0, 100]
        customers = np.clip(customers, 0.0, 100.0)

        demands = self._rng.integers(1, 10, size=n_customers)
        dist_matrix = _compute_distance_matrix(depot, customers)

        return CVRPInstance(
            n_customers=n_customers,
            depot=depot,
            customers=customers,
            demands=demands,
            capacity=capacity,
            distance_matrix=dist_matrix,
            instance_id=f"clustered_{n_customers}_{self.seed}",
        )

    def generate_mixed(
        self, n_customers: int, cluster_ratio: float = 0.6, capacity: int = 50
    ) -> CVRPInstance:
        """Generate a mixed instance: 60% clustered + 40% uniform random.

        Args:
            n_customers: Total number of customer nodes.
            cluster_ratio: Fraction of nodes placed in clusters (default 0.6).
            capacity: Vehicle capacity Q.

        Returns:
            CVRPInstance with mixed layout.
        """
        depot = np.array([50.0, 50.0])

        n_clustered = int(n_customers * cluster_ratio)
        n_random = n_customers - n_clustered

        # Clustered portion
        n_clusters = max(2, n_clustered // 5)
        centres = self._rng.uniform(10.0, 90.0, size=(n_clusters, 2))
        assignments = np.arange(n_clustered) % n_clusters

        clustered_nodes = np.zeros((n_clustered, 2))
        for i in range(n_clustered):
            cluster_id = assignments[i]
            clustered_nodes[i] = self._rng.normal(
                loc=centres[cluster_id], scale=8.0, size=2
            )
        clustered_nodes = np.clip(clustered_nodes, 0.0, 100.0)

        # Random portion
        random_nodes = self._rng.uniform(0.0, 100.0, size=(n_random, 2))

        # Combine and shuffle
        customers = np.vstack([clustered_nodes, random_nodes])
        perm = np.arange(n_customers)
        self._rng.shuffle(perm)
        customers = customers[perm]

        demands = self._rng.integers(1, 10, size=n_customers)
        dist_matrix = _compute_distance_matrix(depot, customers)

        return CVRPInstance(
            n_customers=n_customers,
            depot=depot,
            customers=customers,
            demands=demands,
            capacity=capacity,
            distance_matrix=dist_matrix,
            instance_id=f"mixed_{n_customers}_{self.seed}",
        )

    def load_tsplib(self, filepath: str) -> CVRPInstance:
        """Parse a TSPLIB .vrp file into a CVRPInstance.

        Expected format sections: NAME, DIMENSION, CAPACITY,
        NODE_COORD_SECTION, DEMAND_SECTION, DEPOT_SECTION.

        Args:
            filepath: Path to the .vrp file.

        Returns:
            CVRPInstance parsed from the file.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If required sections are missing.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"TSPLIB file not found: {filepath}")

        lines = path.read_text(encoding="utf-8").strip().splitlines()

        # Parse header
        name = ""
        dimension = 0
        capacity = 0
        coords = {}
        demands_dict = {}
        depot_id = 1

        section = None
        for line in lines:
            line = line.strip()
            if not line or line == "EOF":
                continue

            # Header key-value pairs
            if ":" in line and section is None:
                key, value = line.split(":", 1)
                key = key.strip().upper()
                value = value.strip()
                if key == "NAME":
                    name = value
                elif key == "DIMENSION":
                    dimension = int(value)
                elif key == "CAPACITY":
                    capacity = int(value)
                continue

            # Section markers
            if line.upper() == "NODE_COORD_SECTION":
                section = "coords"
                continue
            elif line.upper() == "DEMAND_SECTION":
                section = "demands"
                continue
            elif line.upper() == "DEPOT_SECTION":
                section = "depot"
                continue

            # Parse section data
            if section == "coords":
                parts = line.split()
                node_id = int(parts[0])
                x, y = float(parts[1]), float(parts[2])
                coords[node_id] = (x, y)
            elif section == "demands":
                parts = line.split()
                node_id = int(parts[0])
                demands_dict[node_id] = int(parts[1])
            elif section == "depot":
                val = int(line)
                if val == -1:
                    section = None
                else:
                    depot_id = val

        if not coords:
            raise ValueError(f"No NODE_COORD_SECTION found in {filepath}")

        # Build arrays (depot = node with depot_id, rest are customers)
        depot_coords = np.array(coords[depot_id], dtype=np.float64)

        customer_ids = sorted([k for k in coords if k != depot_id])
        n_customers = len(customer_ids)

        customers = np.array(
            [coords[cid] for cid in customer_ids], dtype=np.float64
        )
        demands = np.array(
            [demands_dict.get(cid, 1) for cid in customer_ids], dtype=np.int64
        )

        dist_matrix = _compute_distance_matrix(depot_coords, customers)

        return CVRPInstance(
            n_customers=n_customers,
            depot=depot_coords,
            customers=customers,
            demands=demands,
            capacity=capacity if capacity > 0 else 50,
            distance_matrix=dist_matrix,
            instance_id=f"tsplib_{name}_{n_customers}",
        )
