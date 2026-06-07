"""Extracts LP features from a SCIP model state at a B&B node.

Provides utilities to query PySCIPOpt Model objects for variable and 
constraint information used to populate the bipartite graph.
"""

import numpy as np
from pyscipopt import Model


class FeatureExtractor:
    """Extracts LP features and graph structure from a SCIP model.

    Interacts with PySCIPOpt to get current LP values, pseudocosts, 
    and constraint information from the active branching node.
    """

    def __init__(self):
        pass

    def extract_lp_solution(self, model: Model) -> dict[tuple[int, int], float]:
        """Extracts current fractional LP values for all branching variables.
        
        Assumes variables are named 'x_i_j' or similar to map back to edges.
        """
        solution = {}
        vars = model.getVars()
        for var in vars:
            name = var.name
            if name.startswith("x_"):
                # Parse x_i_j
                parts = name.split("_")
                i, j = int(parts[1]), int(parts[2])
                solution[(i, j)] = model.getVal(var)
        return solution

    def extract_pseudocosts(self, model: Model) -> dict[tuple[int, int], tuple[float, float, int]]:
        """Extracts pseudocost information for branching variables."""
        pseudocosts = {}
        vars = model.getVars()
        for var in vars:
            if var.name.startswith("x_"):
                parts = var.name.split("_")
                i, j = int(parts[1]), int(parts[2])
                
                pc_up = model.getVarPseudocost(var, 1) # 1 for UP
                pc_down = model.getVarPseudocost(var, 0) # 0 for DOWN
                # Note: PySCIPOpt might not expose reliability directly in a single call,
                # often it's tracked by SCIP internally. We'll use 1 for now or check SCIP version.
                reliability = 1 
                
                pseudocosts[(i, j)] = (pc_up, pc_down, reliability)
        return pseudocosts

    def extract_constraints(self, model: Model) -> list[dict]:
        """Extracts constraint features from the SCIP model.
        
        Captures RHS, activity, slack, and constraint type.
        """
        constraints = []
        conss = model.getConss()
        
        for cons in conss:
            rhs = model.getRHS(cons)
            activity = model.getActivity(cons)
            slack = rhs - activity
            
            # Simple heuristic for type mapping
            # In a real impl, we'd check the constraint handler name
            c_type = 0 # Default: Capacity
            if "visit" in cons.name:
                c_type = 1 # Covering
            elif "subtour" in cons.name or "sec" in cons.name:
                c_type = 2 # Subtour
                
            # Find which variables are in this constraint
            # This is expensive in SCIP but necessary for the bipartite graph
            involved_vars = []
            # Note: This usually requires specific SCIP handler access or row inspection
            # For the demo/workflow, we'll return a placeholder or implement basic row scan
            
            constraints.append({
                "rhs": float(rhs) if rhs < 1e20 else 1.0,
                "activity": float(activity),
                "slack": float(slack),
                "type": c_type,
                "involved_edges": [] # To be filled by model-specific logic
            })
            
        return constraints
