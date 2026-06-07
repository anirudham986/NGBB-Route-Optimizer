"""Records B&B solve events for training data and visualization.

Attaches to SCIP as an event handler to capture branching decisions,
pruning, incumbent updates, and solve completion.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import pyscipopt


@dataclass
class SolveEvent:
    """A single event from the B&B solve process."""
    frame: int
    event_type: str  # 'branch' | 'prune' | 'incumbent' | 'solved'
    node_id: int
    depth: int
    branch_edge: Optional[tuple[int, int]]
    lower_bound: float
    upper_bound: float
    nodes_explored: int
    active_edges: list[tuple[int, int]] = field(default_factory=list)
    pruned_nodes: list[int] = field(default_factory=list)


class EventLogger(pyscipopt.Eventhdlr):
    """SCIP event handler that records solve events."""

    def __init__(self):
        super().__init__()
        self.events: list[SolveEvent] = []
        self._frame = 0

    def eventinit(self):
        """Register for relevant SCIP events."""
        self.model.catchEvent(
            pyscipopt.SCIP_EVENTTYPE.NODEFOCUSED
            | pyscipopt.SCIP_EVENTTYPE.NODEFEASIBLE
            | pyscipopt.SCIP_EVENTTYPE.NODEINFEASIBLE
            | pyscipopt.SCIP_EVENTTYPE.BESTSOLFOUND,
            self,
        )

    def eventexit(self):
        self.model.dropEvent(
            pyscipopt.SCIP_EVENTTYPE.NODEFOCUSED
            | pyscipopt.SCIP_EVENTTYPE.NODEFEASIBLE
            | pyscipopt.SCIP_EVENTTYPE.NODEINFEASIBLE
            | pyscipopt.SCIP_EVENTTYPE.BESTSOLFOUND,
            self,
        )

    def eventexec(self, event):
        """Called by SCIP on each registered event."""
        etype = event.getType()
        node = self.model.getCurrentNode()
        node_id = node.getNumber() if node else -1
        depth = node.getDepth() if node else 0

        lb = self.model.getDualbound()
        ub = self.model.getPrimalbound()
        n_nodes = self.model.getNNodes()

        if etype & pyscipopt.SCIP_EVENTTYPE.BESTSOLFOUND:
            evt_type = "incumbent"
        elif etype & pyscipopt.SCIP_EVENTTYPE.NODEINFEASIBLE:
            evt_type = "prune"
        elif etype & pyscipopt.SCIP_EVENTTYPE.NODEFEASIBLE:
            evt_type = "prune"
        else:
            evt_type = "branch"

        self.events.append(SolveEvent(
            frame=self._frame,
            event_type=evt_type,
            node_id=node_id,
            depth=depth,
            branch_edge=None,
            lower_bound=lb,
            upper_bound=ub if ub < 1e20 else float('inf'),
            nodes_explored=n_nodes,
        ))
        self._frame += 1

    def add_solved_event(self):
        """Manually add a terminal 'solved' event after optimization."""
        lb = self.model.getDualbound()
        ub = self.model.getPrimalbound()
        self.events.append(SolveEvent(
            frame=self._frame,
            event_type="solved",
            node_id=-1,
            depth=0,
            branch_edge=None,
            lower_bound=lb,
            upper_bound=ub if ub < 1e20 else float('inf'),
            nodes_explored=self.model.getNNodes(),
        ))

    def get_events(self) -> list[SolveEvent]:
        return self.events

    def save_events(self, filepath: str) -> None:
        """Save event log to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(e) for e in self.events]
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def load_events(filepath: str) -> list[SolveEvent]:
        """Load event log from JSON."""
        data = json.loads(Path(filepath).read_text(encoding="utf-8"))
        return [SolveEvent(**e) for e in data]
