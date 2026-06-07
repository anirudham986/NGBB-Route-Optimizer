/**
 * Derives display state from a solver result at a given frame index.
 */
export function deriveFrameState(result, frame) {
  if (!result || !result.events || result.events.length === 0) {
    return {
      currentEvent: null, nodesExplored: 0, bestCost: null,
      lowerBound: 0, gap: null, visitedNodes: new Set(),
      activeEdges: [], candidateEdge: null, prunedNodes: new Set(),
      isSolved: false, optimalRoute: null, simulatedTimeMs: 0, frame: 0,
    };
  }

  const clampedFrame = Math.min(frame, result.events.length - 1);
  const event = result.events[clampedFrame];
  const prevEvents = result.events.slice(0, clampedFrame + 1);

  return {
    currentEvent: event,
    nodesExplored: event.nodesExplored,
    bestCost: event.upperBound === Infinity ? null : event.upperBound,
    lowerBound: event.lowerBound,
    gap: event.upperBound === Infinity ? null
         : Math.abs(event.upperBound - result.optimalCost) / Math.max(result.optimalCost, 0.001) * 100,
    visitedNodes: new Set(prevEvents.map(e => e.nodeId)),
    activeEdges: event.activeEdges || [],
    candidateEdge: event.type === 'branch' ? event.branchEdge : null,
    prunedNodes: new Set(prevEvents.flatMap(e => e.prunedNodes || [])),
    isSolved: event.type === 'solved',
    optimalRoute: event.type === 'solved' ? result.optimalRoute : null,
    simulatedTimeMs: Math.round(event.nodesExplored * 2.4),
    frame: clampedFrame,
  };
}
