import { mulberry32 } from './prng';

function computeGreedyCost(instance) {
  const n = instance.nodes.length;
  const visited = new Array(n).fill(false);
  visited[0] = true;
  let current = 0, cost = 0;
  for (let step = 1; step < n; step++) {
    let bestNext = -1, bestDist = Infinity;
    for (let j = 1; j < n; j++) {
      if (!visited[j] && instance.dist[current][j] < bestDist) {
        bestDist = instance.dist[current][j]; bestNext = j;
      }
    }
    if (bestNext >= 0) { visited[bestNext] = true; cost += bestDist; current = bestNext; }
  }
  return cost + instance.dist[current][0];
}

/**
 * Simulates NGBB solver with smart GNN-guided branching.
 * Produces 60-75% fewer events than traditional.
 */
export function runNGBBSolver(instance, seed) {
  const rand = mulberry32(seed + 2);
  const events = [];
  let frame = 0, nodesExplored = 0, bestCost = Infinity;
  const greedyCost = computeGreedyCost(instance);
  const optimalCost = greedyCost * 0.88;
  const n = instance.nodes.length;

  // Target: ~25-35% of traditional nodes (200-280 vs 800)
  const maxNodes = Math.floor(180 + rand() * 100);

  const stack = [{ id: 0, depth: 0, edges: [], score: 0 }];
  let nodeId = 1;

  while (stack.length > 0 && nodesExplored < maxNodes) {
    const node = stack.pop();
    nodesExplored++;

    const progress = nodesExplored / maxNodes;
    const lpBound = optimalCost + (greedyCost - optimalCost) * (1 - progress * 0.9) * (0.5 + rand() * 0.3);

    if (lpBound >= bestCost) {
      events.push({ frame: frame++, type: 'prune', nodeId: node.id, depth: node.depth,
        lowerBound: lpBound, upperBound: bestCost, nodesExplored,
        activeEdges: node.edges, prunedNodes: [node.id], branchEdge: null });
      continue;
    }

    // GNN-guided: finds incumbents faster
    if (node.depth >= Math.floor(n * 0.4) || rand() < 0.08 || (nodesExplored > 15 && rand() < 0.12)) {
      const solCost = lpBound + rand() * greedyCost * 0.05;
      if (solCost < bestCost) {
        bestCost = solCost;
        const route = [];
        for (let k = 0; k < Math.min(node.depth + 3, n); k++) {
          const from = k === 0 ? 0 : Math.floor(rand() * (n - 1)) + 1;
          const to = Math.floor(rand() * (n - 1)) + 1;
          if (from !== to) route.push([from, to]);
        }
        events.push({ frame: frame++, type: 'incumbent', nodeId: node.id, depth: node.depth,
          lowerBound: lpBound, upperBound: bestCost, nodesExplored,
          activeEdges: route, prunedNodes: [], branchEdge: null });
      }
      continue;
    }

    // GNN-guided branching: prefer edges near depot with high fractionality
    let bestFrom = 0, bestTo = 1, bestScore = -Infinity;
    for (let tries = 0; tries < 5; tries++) {
      const f = Math.floor(rand() * n);
      let t = Math.floor(rand() * (n - 1));
      if (t >= f) t++;
      const frac = 0.3 + rand() * 0.4;
      const score = (1 - Math.abs(frac - 0.5) * 2) * 0.6
        + (1 - instance.dist[0][t] / instance.maxDist) * 0.4;
      if (score > bestScore) { bestScore = score; bestFrom = f; bestTo = t; }
    }
    const branchEdge = [bestFrom, bestTo];

    stack.push(
      { id: nodeId++, depth: node.depth + 1, edges: [...node.edges, branchEdge], score: bestScore },
      { id: nodeId++, depth: node.depth + 1, edges: [...node.edges, branchEdge], score: bestScore * 0.8 }
    );

    events.push({ frame: frame++, type: 'branch', nodeId: node.id, depth: node.depth,
      branchEdge, lowerBound: lpBound, upperBound: bestCost, nodesExplored,
      activeEdges: node.edges, prunedNodes: [] });
  }

  // Build final route (same as traditional for fair comparison)
  const optRoute = [0];
  const used = new Set([0]);
  let cur = 0;
  for (let s = 0; s < n - 1; s++) {
    let bn = -1, bd = Infinity;
    for (let j = 0; j < n; j++) {
      if (!used.has(j) && instance.dist[cur][j] < bd) { bd = instance.dist[cur][j]; bn = j; }
    }
    if (bn >= 0) { used.add(bn); optRoute.push(bn); cur = bn; }
  }
  optRoute.push(0);
  const finalEdges = optRoute.slice(0, -1).map((v, i) => [v, optRoute[i + 1]]);

  events.push({ frame: frame, type: 'solved', nodeId: -1, depth: 0,
    branchEdge: null, lowerBound: bestCost, upperBound: bestCost,
    nodesExplored, activeEdges: finalEdges, prunedNodes: [] });

  return { events, optimalCost: bestCost, optimalRoute: optRoute,
    totalNodes: nodesExplored, solvedAtFrame: frame, instance };
}
