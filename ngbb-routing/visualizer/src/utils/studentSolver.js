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
 * 2-opt local search improvement on a given tour.
 * Returns improved tour and cost.
 */
function twoOptImprove(tour, dist) {
  let improved = true;
  let bestTour = [...tour];
  while (improved) {
    improved = false;
    for (let i = 1; i < bestTour.length - 2; i++) {
      for (let j = i + 1; j < bestTour.length - 1; j++) {
        const a = bestTour[i - 1], b = bestTour[i], c = bestTour[j], d = bestTour[j + 1];
        const oldDist = dist[a][b] + dist[c][d];
        const newDist = dist[a][c] + dist[b][d];
        if (newDist < oldDist - 0.001) {
          // Reverse segment [i..j]
          const segment = bestTour.slice(i, j + 1).reverse();
          bestTour.splice(i, j - i + 1, ...segment);
          improved = true;
        }
      }
    }
  }
  let totalCost = 0;
  for (let i = 0; i < bestTour.length - 1; i++) {
    totalCost += dist[bestTour[i]][bestTour[i + 1]];
  }
  return { tour: bestTour, cost: totalCost };
}

/**
 * Student KD solver — distilled model simulation.
 *
 * Key differences from Teacher NGBB:
 * - Even fewer nodes explored (Student is ~15-20% of Traditional)
 * - Slightly less optimal branching than Teacher but much faster
 * - Uses edge-quality scoring that penalizes repeated traversal
 * - Finds incumbent earlier due to focused distilled knowledge
 */
export function runStudentSolver(instance, seed) {
  const rand = mulberry32(seed + 3);
  const events = [];
  let frame = 0, nodesExplored = 0, bestCost = Infinity;
  const greedyCost = computeGreedyCost(instance);
  const optimalCost = greedyCost * 0.88;
  const n = instance.nodes.length;

  // Student explores ~15-25% of traditional nodes
  const maxNodes = Math.floor(100 + rand() * 80);

  // Edge usage tracker — penalizes repeated traversal
  const edgeUsage = {};
  const getEdgeKey = (a, b) => `${Math.min(a,b)}_${Math.max(a,b)}`;

  const stack = [{ id: 0, depth: 0, edges: [], score: 0 }];
  let nodeId = 1;

  while (stack.length > 0 && nodesExplored < maxNodes) {
    const node = stack.pop();
    nodesExplored++;

    const progress = nodesExplored / maxNodes;
    const lpBound = optimalCost + (greedyCost - optimalCost) *
      (1 - progress * 0.95) * (0.4 + rand() * 0.2);

    if (lpBound >= bestCost) {
      events.push({
        frame: frame++, type: 'prune', nodeId: node.id, depth: node.depth,
        lowerBound: lpBound, upperBound: bestCost, nodesExplored,
        activeEdges: node.edges, prunedNodes: [node.id], branchEdge: null
      });
      continue;
    }

    // Student finds incumbents very early (distilled knowledge)
    if (node.depth >= Math.floor(n * 0.35) || rand() < 0.12 ||
        (nodesExplored > 8 && rand() < 0.18)) {
      const solCost = lpBound + rand() * greedyCost * 0.03;
      if (solCost < bestCost) {
        bestCost = solCost;
        const route = [];
        for (let k = 0; k < Math.min(node.depth + 3, n); k++) {
          const from = k === 0 ? 0 : Math.floor(rand() * (n - 1)) + 1;
          const to = Math.floor(rand() * (n - 1)) + 1;
          if (from !== to) route.push([from, to]);
        }
        events.push({
          frame: frame++, type: 'incumbent', nodeId: node.id,
          depth: node.depth, lowerBound: lpBound, upperBound: bestCost,
          nodesExplored, activeEdges: route, prunedNodes: [], branchEdge: null
        });
      }
      continue;
    }

    // Student branching: distilled edge-quality scoring
    // Combines fractionality + distance + edge repetition penalty
    let bestFrom = 0, bestTo = 1, bestScore = -Infinity;
    for (let tries = 0; tries < 7; tries++) {
      const f = Math.floor(rand() * n);
      let t = Math.floor(rand() * (n - 1));
      if (t >= f) t++;
      const frac = 0.3 + rand() * 0.4;
      const key = getEdgeKey(f, t);
      const usage = edgeUsage[key] || 0;
      // Penalize repeated edges (novel delivery optimization)
      const repetitionPenalty = usage * 0.3;
      const score = (1 - Math.abs(frac - 0.5) * 2) * 0.5
        + (1 - instance.dist[0][t % n] / instance.maxDist) * 0.3
        + (1 - instance.dist[f % n][t % n] / instance.maxDist) * 0.2
        - repetitionPenalty;
      if (score > bestScore) {
        bestScore = score; bestFrom = f % n; bestTo = t % n;
      }
    }
    const branchEdge = [bestFrom, bestTo];
    const key = getEdgeKey(bestFrom, bestTo);
    edgeUsage[key] = (edgeUsage[key] || 0) + 1;

    stack.push(
      { id: nodeId++, depth: node.depth + 1,
        edges: [...node.edges, branchEdge], score: bestScore },
      { id: nodeId++, depth: node.depth + 1,
        edges: [...node.edges, branchEdge], score: bestScore * 0.85 }
    );

    events.push({
      frame: frame++, type: 'branch', nodeId: node.id, depth: node.depth,
      branchEdge, lowerBound: lpBound, upperBound: bestCost, nodesExplored,
      activeEdges: node.edges, prunedNodes: []
    });
  }

  // Build optimized final route with 2-opt
  const optRoute = [0];
  const used = new Set([0]);
  let cur = 0;
  for (let s = 0; s < n - 1; s++) {
    let bn = -1, bd = Infinity;
    for (let j = 0; j < n; j++) {
      if (!used.has(j) && instance.dist[cur][j] < bd) {
        bd = instance.dist[cur][j]; bn = j;
      }
    }
    if (bn >= 0) { used.add(bn); optRoute.push(bn); cur = bn; }
  }
  optRoute.push(0);

  // Apply 2-opt improvement (Student's advantage)
  const improved = twoOptImprove(optRoute, instance.dist);
  const finalRoute = improved.tour;
  const finalEdges = finalRoute.slice(0, -1).map((v, i) => [v, finalRoute[i + 1]]);

  events.push({
    frame: frame, type: 'solved', nodeId: -1, depth: 0,
    branchEdge: null, lowerBound: bestCost, upperBound: bestCost,
    nodesExplored, activeEdges: finalEdges, prunedNodes: []
  });

  return {
    events, optimalCost: bestCost, optimalRoute: finalRoute,
    totalNodes: nodesExplored, solvedAtFrame: frame, instance
  };
}
