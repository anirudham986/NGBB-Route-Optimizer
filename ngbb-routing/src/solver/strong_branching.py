"""Strong branching oracle for training data collection.

Evaluates all candidate variables by actually solving both child LPs,
then selects the variable with the highest bound improvement. Records
(graph_state, label) pairs for imitation learning.
"""

import torch
import pyscipopt
from dataclasses import dataclass

from src.data.feature_extractor import FeatureExtractor
from src.data.graph_constructor import construct_bipartite_graph


@dataclass
class BranchingSample:
    """A single training sample: graph state + oracle label."""
    graph: object  # HeteroData
    label: int     # index of chosen variable among candidates
    score: float   # strong branching score of chosen variable


class StrongBranchingCollector(pyscipopt.Branchrule):
    """SCIP branching rule that performs full strong branching and logs data."""

    def __init__(self, instance, extractor: FeatureExtractor, node_limit: int = 500):
        super().__init__()
        self.instance = instance
        self.extractor = extractor
        self.node_limit = node_limit
        self.samples = []
        self.nodes_processed = 0
        self.name = "StrongBranchCollector"
        self.desc = "Full strong branching oracle for data collection"
        self.priority = 100000
        self.maxdepth = -1
        self.maxbounddist = 1.0

    def branchexeclp(self, allowaddcons):
        """Perform strong branching at each node and record the decision."""
        if self.nodes_processed >= self.node_limit:
            return {"result": pyscipopt.SCIP_RESULT.DIDNOTRUN}

        candidates, _, _, ncands = self.model.getLPBranchCands()
        if ncands == 0:
            return {"result": pyscipopt.SCIP_RESULT.DIDNOTRUN}

        # Extract current state as bipartite graph
        lp_sol = self.extractor.extract_lp_solution(self.model)
        graph = construct_bipartite_graph(self.instance, lp_sol)

        # Evaluate each candidate via strong branching
        best_idx = 0
        best_score = -float('inf')
        scores = []

        for i, var in enumerate(candidates):
            # SCIP's strong branching: solve both child LPs
            down_obj, up_obj, _, _, _, _ = self.model.getVarStrongbranch(var)

            # Score = product scoring (Khalil et al. 2016)
            current_obj = self.model.getLPObjVal()
            down_gain = max(down_obj - current_obj, 1e-6)
            up_gain = max(up_obj - current_obj, 1e-6)
            sb_score = (1 - 0.2) * min(down_gain, up_gain) + 0.2 * max(down_gain, up_gain)
            scores.append(sb_score)

            if sb_score > best_score:
                best_score = sb_score
                best_idx = i

        # Record sample
        self.samples.append(BranchingSample(
            graph=graph,
            label=best_idx,
            score=best_score,
        ))

        # Branch on best variable
        self.model.branchVar(candidates[best_idx])
        self.nodes_processed += 1

        return {"result": pyscipopt.SCIP_RESULT.BRANCHED}

    def get_samples(self) -> list[BranchingSample]:
        """Return all collected (graph, label) samples."""
        return self.samples
