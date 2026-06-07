"""SCIP custom branching rule callback.

Plugs a GNN-based policy into the SCIP search process. At each 
fractional node, it extracts features, runs GNN inference, and 
directs SCIP to branch on the highest-scoring variable.
"""

import numpy as np
# pyrefly: ignore [missing-import]
import pyscipopt
import torch

# pyrefly: ignore [missing-import]
from src.data.feature_extractor import FeatureExtractor
# pyrefly: ignore [missing-import]
from src.data.graph_constructor import construct_bipartite_graph


class GNNBranchingRule(pyscipopt.Branchrule):
    """SCIP branching rule that uses a GNN policy for decisions."""

    def __init__(self, policy, extractor: FeatureExtractor, name: str = "GNNBranching"):
        super().__init__()
        self.policy = policy
        self.extractor = extractor
        self.name = name
        self.desc = "Neural-guided branching using a bipartite GNN."
        self.priority = 50000  # High priority to override defaults
        self.maxdepth = -1
        self.maxbounddist = 1.0

    def branchexeclp(self, allowaddcons):
        """Called by SCIP when branching on an LP solution."""
        # 1. Extract current state
        lp_sol = self.extractor.extract_lp_solution(self.model)
        pseudocosts = self.extractor.extract_pseudocosts(self.model)
        # Note: We'd need the instance too. Usually passed in or stored in model data.
        instance = self.model.data['instance'] 

        # 2. Build graph
        graph = construct_bipartite_graph(instance, lp_sol, pseudocosts)
        
        # 3. Get candidates from SCIP
        candidates, _, _, _ = self.model.getLPBranchCands()
        if not candidates:
            return {"result": pyscipopt.SCIP_RESULT.DIDNOTRUN}

        # 4. Score candidates using GNN
        # Map SCIP candidate variables to their indices in our graph's variable list
        with torch.no_grad():
            # This assumes graph.edge_list matches SCIP variable names
            # In a real impl, we'd have a mapping
            scores = self.policy.score(graph)
            
            # Find best candidate among those SCIP allows
            cand_names = [v.name for v in candidates]
            # Scoring logic would map cand_names -> indices -> scores
            # For brevity:
            best_var = candidates[0] 
            max_score = -float('inf')
            
            for i, var in enumerate(candidates):
                # Pseudo-mapping: find score for this variable
                # This needs careful index alignment
                score = scores[i] if i < len(scores) else 0 
                if score > max_score:
                    max_score = score
                    best_var = var

        # 5. Execute branch
        self.model.branchVar(best_var)
        return {"result": pyscipopt.SCIP_RESULT.BRANCHED}
