import { mulberry32 } from './prng';

/**
 * Generate a CVRP/TSP instance for visualization.
 */
export function generateInstance(nodeCount, seed, problemType, vehicleCapacity = 50) {
  const rand = mulberry32(seed);

  const depot = { x: 50, y: 50, id: 0, isDepot: true, demand: 0 };
  const customers = Array.from({ length: nodeCount }, (_, i) => ({
    x: rand() * 90 + 5,
    y: rand() * 90 + 5,
    id: i + 1,
    isDepot: false,
    demand: Math.floor(rand() * 9) + 1,
  }));
  const nodes = [depot, ...customers];

  // Distance matrix
  const dist = Array.from({ length: nodes.length }, (_, i) =>
    Array.from({ length: nodes.length }, (_, j) => {
      const dx = nodes[i].x - nodes[j].x;
      const dy = nodes[i].y - nodes[j].y;
      return Math.sqrt(dx * dx + dy * dy);
    })
  );

  const maxDist = Math.max(...dist.flat());

  return { nodes, dist, depot, customers, vehicleCapacity, problemType, maxDist };
}
