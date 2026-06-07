import { mulberry32 } from './prng';

/**
 * Compute a greedy nearest-neighbour tour cost.
 */
function computeGreedyCost(instance) {
  const n = instance.nodes.length;
  const visited = new Array(n).fill(false);
  visited[0] = true;
  let current = 0, cost = 0;
  for (let step = 1; step < n; step++) {
    let bestNext = -1, bestDist = Infinity;
    for (let j = 1; j < n; j++) {
      if (!visited[j] && instance.dist[current][j] < bestDist) {
        bestDist = instance.dist[current][j];
        bestNext = j;
      }
    }
    if (bestNext >= 0) { visited[bestNext] = true; cost += bestDist; current = bestNext; }
  }
  return cost + instance.dist[current][0];
}

function createBBNode(id, depth, edges, fixedVars) {
  return { id, depth, edges: [...edges], fixedVars: { ...fixedVars } };
}

/**
 * Simulates vanilla B&B with random branching (poor heuristic).
 * Produces a long event log — ~400-800 nodes explored.
 */
export function runTraditionalSolver(instance, seed) {
  const rand = mulberry32(seed + 1);
  const events = [];
  let frame = 0, nodesExplored = 0, bestCost = Infinity;
  const greedyCost = computeGreedyCost(instance);
  const optimalCost = greedyCost * 0.88;
  const n = instance.nodes.length;

  // Simulated DFS B&B
  const stack = [createBBNode(0, 0, [], {})];
  let nodeIdCounter = 1;
  const allPruned = [];

  while (stack.length > 0 && nodesExplored < 800) {
    const node = stack.pop();
    nodesExplored++;

    // Simulate LP bound
    const progress = nodesExplored / 800;
    const noise = (rand() - 0.5) * greedyCost * 0.3;
    const lpBound = optimalCost + (greedyCost - optimalCost) * (1 - progress * 0.7) + noise;

    if (lpBound >= bestCost) {
      allPruned.push(node.id);
      events.push({ frame: frame++, type: 'prune', nodeId: node.id, depth: node.depth,
        lowerBound: lpBound, upperBound: bestCost, nodesExplored,
        activeEdges: node.edges, prunedNodes: [node.id], branchEdge: null });
      continue;
    }

    // Check integer feasibility (simulated)
    if (node.depth >= Math.floor(n * 0.6) || rand() < 0.04) {
      const solCost = lpBound + rand() * greedyCost * 0.1;
      if (solCost < bestCost) {
        bestCost = solCost;
        // Build a partial route for visualization
        const route = [];
        for (let k = 0; k < Math.min(node.depth + 2, n); k++) {
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

    // Branch: random edge selection (bad heuristic)
    const from = Math.floor(rand() * n);
    let to = Math.floor(rand() * (n - 1));
    if (to >= from) to++;
    const branchEdge = [from, to];

    const child0 = createBBNode(nodeIdCounter++, node.depth + 1,
      [...node.edges, branchEdge], { ...node.fixedVars, [`${from}_${to}`]: 0 });
    const child1 = createBBNode(nodeIdCounter++, node.depth + 1,
      [...node.edges, branchEdge], { ...node.fixedVars, [`${from}_${to}`]: 1 });
    stack.push(child1, child0);

    events.push({ frame: frame++, type: 'branch', nodeId: node.id, depth: node.depth,
      branchEdge, lowerBound: lpBound, upperBound: bestCost, nodesExplored,
      activeEdges: node.edges, prunedNodes: [] });
  }

  // Build final optimal route
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
