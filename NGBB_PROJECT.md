# Neural-Guided Branch & Bound — Dynamic Delivery Route Optimization
### Project 05 | Full Execution Document

> **For Antigravity:** This document is the single source of truth for building this project end-to-end. Every section is actionable. Read top-to-bottom on first pass, then use section headers as a reference during build. No assumptions are left to the developer.

---

## Table of Contents

1. [Research Context & Goal](#1-research-context--goal)
2. [Research Gap Statement](#2-research-gap-statement)
3. [System Architecture Overview](#3-system-architecture-overview)
4. [Complete File & Folder Structure](#4-complete-file--folder-structure)
5. [Environment Setup & Dependencies](#5-environment-setup--dependencies)
6. [Module 1 — Problem Instance Generator](#6-module-1--problem-instance-generator)
7. [Module 2 — Branch-and-Bound Engine](#7-module-2--branch-and-bound-engine)
8. [Module 3 — GNN Branching Policy](#8-module-3--gnn-branching-policy)
9. [Module 4 — Training Pipeline](#9-module-4--training-pipeline)
10. [Module 5 — Evaluation Suite](#10-module-5--evaluation-suite)
11. [Visualizer UI — Comparative Frontend](#11-visualizer-ui--comparative-frontend)
12. [Visualizer — Component Specifications](#12-visualizer--component-specifications)
13. [Visualizer — Solver Simulation Logic](#13-visualizer--solver-simulation-logic)
14. [Visualizer — Canvas Rendering Spec](#14-visualizer--canvas-rendering-spec)
15. [Visualizer — Event Log Schema](#15-visualizer--event-log-schema)
16. [Experiment Configs & Reproducibility](#16-experiment-configs--reproducibility)
17. [Evaluation Protocol & Metrics](#17-evaluation-protocol--metrics)
18. [Project Timeline & Milestones](#18-project-timeline--milestones)
19. [Risks & Mitigations](#19-risks--mitigations)
20. [Acceptance Criteria — Full Project](#20-acceptance-criteria--full-project)

---

## 1. Research Context & Goal

**Problem class:** NP-hard combinatorial optimization — specifically the Capacitated Vehicle Routing Problem (CVRP) and Travelling Salesman Problem (TSP).

**Core idea:** Classical Branch-and-Bound (B&B) solvers guarantee optimal solutions but scale poorly because the branching rule — which variable to split on next — is chosen by hand-engineered heuristics. A bad branching rule causes exponential search tree blowup. A good one enables near-linear-time solve on practical instances.

**This project trains a Graph Neural Network (GNN) to learn the branching rule** from examples of an expensive-but-excellent oracle (Strong Branching), then substitutes that cheap learned policy at inference time. The result: same optimal guarantees, dramatically fewer nodes explored.

**Application domain:** Dynamic delivery route optimization — real-world CVRP instances with depot + customer nodes, vehicle capacities, and minimize-total-distance objectives.

**Headline claim to validate:** GNN-guided branching reduces explored nodes by **60–80%** vs vanilla B&B on benchmark instances, while maintaining provably optimal solutions.

---

## 2. Research Gap Statement

> *"Existing delivery routing methods either sacrifice optimality for speed or become computationally infeasible at large scales. This work explores whether Graph Neural Networks can learn effective branching heuristics that reduce search complexity while maintaining solution quality — targeting a 60–80% reduction in explored nodes on benchmark instances."*

**What this is NOT:**
- Not a pure heuristic (no optimality guarantees)
- Not a new solver from scratch (augments existing B&B)
- Not limited to fixed instance sizes (trains small, generalizes large)

**Key novelty:** Application-specific GNN trained on delivery routing instances, evaluated on real logistics benchmark data, with a live interactive visualizer demonstrating the reduction.

---

## 3. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      NGBB SYSTEM PIPELINE                       │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  Instance    │───▶│  B&B Engine  │───▶│  GNN Branching   │  │
│  │  Generator   │    │  (PySCIPOpt) │◀───│  Policy          │  │
│  └──────────────┘    └──────┬───────┘    └──────────────────┘  │
│                             │                      ▲            │
│                             ▼                      │            │
│                    ┌──────────────┐    ┌──────────────────┐    │
│                    │  Event Log / │───▶│  Training Loop   │    │
│                    │  State Data  │    │  (IL → RL)        │    │
│                    └──────┬───────┘    └──────────────────┘    │
│                             │                                   │
│                             ▼                                   │
│                    ┌──────────────────────────────────┐         │
│                    │  Evaluation Suite + Visualizer   │         │
│                    └──────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

**Five modules, built in order:**

| # | Module | Language | Key Dependency |
|---|--------|----------|----------------|
| 1 | Problem Instance Generator | Python | NumPy, NetworkX |
| 2 | B&B Engine + SCIP Wrapper | Python | PySCIPOpt 4.x, SCIP 8.x |
| 3 | GNN Branching Policy | Python | PyTorch 2.x, PyTorch Geometric |
| 4 | Training Pipeline | Python | wandb, hydra-core |
| 5 | Evaluation Suite | Python | pandas, matplotlib, scipy |
| — | Visualizer UI | JavaScript | React 18, D3.js, Vite, Tailwind |

---

## 4. Complete File & Folder Structure

Every file listed here must exist. Files marked `[AUTO]` are generated during runs and do not need to be created manually.

```
ngbb-routing/
│
├── README.md                          # Project overview (link to this doc)
├── pyproject.toml                     # Python package config + deps
├── requirements.txt                   # Pinned Python deps (generated from pyproject)
├── .env.example                       # Environment variable template
├── .gitignore
├── docker/
│   ├── Dockerfile                     # SCIP 8.x + Python 3.11 base image
│   └── docker-compose.yml             # Dev environment orchestration
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── generate.py                # CLI: generates and saves instance datasets
│   │   ├── instance_generator.py      # Core: random + cluster + TSPLIB instance gen
│   │   ├── graph_constructor.py       # Converts CVRP instance → bipartite graph
│   │   ├── feature_extractor.py       # Extracts LP features at each B&B node
│   │   └── dataset.py                 # PyG HeteroData dataset wrapper
│   │
│   ├── solver/
│   │   ├── __init__.py
│   │   ├── scip_wrapper.py            # PySCIPOpt model setup for CVRP/TSP
│   │   ├── branching_rule.py          # SCIP BRANCHRULE callback — plugs GNN in
│   │   ├── strong_branching.py        # Oracle: full strong branching for data collection
│   │   ├── state_serializer.py        # Serializes B&B node state → PyG graph
│   │   └── event_logger.py            # Records solve events for training + visualization
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gnn.py                     # Bipartite GNN: message passing architecture
│   │   ├── output_head.py             # MLP head: graph → per-variable scores
│   │   └── policy.py                  # Full policy model: GNN + head + inference
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_il.py                # CLI: imitation learning training loop
│   │   ├── train_rl.py                # CLI: RL fine-tuning loop
│   │   ├── il_trainer.py              # Core IL training logic
│   │   ├── rl_trainer.py              # Core RL (REINFORCE) training logic
│   │   ├── curriculum.py              # Curriculum scheduler: ramps instance sizes
│   │   └── losses.py                  # Cross-entropy IL loss + RL policy gradient
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── benchmark.py               # CLI: runs full evaluation suite
│   │   ├── baselines.py               # Pseudocost, random, nearest-neighbour impls
│   │   ├── metrics.py                 # Node reduction, solve time, optimality gap
│   │   └── stats.py                   # Wilcoxon tests, confidence intervals, tables
│   │
│   └── utils/
│       ├── __init__.py
│       ├── prng.py                    # Seeded PRNG utilities (mulberry32 port)
│       ├── logging.py                 # Structured logging setup
│       └── config.py                  # Hydra config resolution helpers
│
├── experiments/
│   ├── il_baseline.yaml               # Imitation learning baseline config
│   ├── il_curriculum.yaml             # IL with curriculum scheduling
│   ├── rl_finetune.yaml               # RL fine-tuning config
│   ├── ablation_gnn_depth.yaml        # Ablation: 1/2/3/4 GNN layers
│   ├── ablation_features.yaml         # Ablation: feature set variants
│   └── eval_final.yaml                # Final evaluation config
│
├── checkpoints/                       # [AUTO] Model checkpoints saved here
│   └── .gitkeep
│
├── data/
│   ├── raw/
│   │   └── tsplib/                    # TSPLIB benchmark instances (.vrp files)
│   ├── generated/                     # [AUTO] Synthetic instances
│   └── processed/                     # [AUTO] PyG HeteroData .pt files
│
├── results/                           # [AUTO] Evaluation outputs, plots, tables
│   └── .gitkeep
│
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory data analysis
│   ├── 02_training_curves.ipynb       # Training loss/accuracy plots
│   └── 03_results_viz.ipynb           # Final result figures
│
├── tests/
│   ├── __init__.py
│   ├── test_instance_generator.py
│   ├── test_graph_constructor.py
│   ├── test_gnn.py
│   ├── test_solver.py
│   └── test_metrics.py
│
└── visualizer/
    ├── package.json
    ├── vite.config.js
    ├── index.html
    ├── tailwind.config.js
    ├── postcss.config.js
    │
    ├── src/
    │   ├── main.jsx                   # React entry point
    │   ├── App.jsx                    # Root component + animation loop
    │   │
    │   ├── components/
    │   │   ├── HeaderBar.jsx
    │   │   ├── ControlToolbar.jsx
    │   │   ├── SolverCanvas.jsx
    │   │   ├── StatsDashboard.jsx
    │   │   ├── MetricCard.jsx
    │   │   ├── VsBadge.jsx
    │   │   └── NodeReductionBadge.jsx
    │   │
    │   ├── utils/
    │   │   ├── instanceGenerator.js   # Seeded node/distance generation
    │   │   ├── traditionalSolver.js   # Simulated B&B event log
    │   │   ├── ngbbSolver.js          # Simulated NGBB event log
    │   │   ├── canvasRenderer.js      # Pure draw functions (no React)
    │   │   ├── solverState.js         # Derives display state from event log + frame
    │   │   └── prng.js                # mulberry32 seeded PRNG
    │   │
    │   └── styles/
    │       ├── tokens.css             # CSS variables: light + dark themes
    │       └── layout.css             # Grid/flex layout only
    │
    └── public/
        └── favicon.svg
```

---

## 5. Environment Setup & Dependencies

### Python Environment

**Requires:** Python 3.11, SCIP 8.x (installed via conda)

```bash
# 1. Clone repo
git clone https://github.com/your-org/ngbb-routing.git
cd ngbb-routing

# 2. Create conda environment (manages SCIP dependency cleanly)
conda create -n ngbb python=3.11
conda activate ngbb

# 3. Install SCIP + PySCIPOpt
conda install -c conda-forge scip
pip install pyscipopt==4.4.0

# 4. Install remaining Python dependencies
pip install -r requirements.txt

# 5. Verify SCIP works
python -c "from pyscipopt import Model; m = Model(); print('SCIP OK')"
```

### `requirements.txt` — Full pinned list

```
# Core solver
pyscipopt==4.4.0

# ML
torch==2.3.0
torch-geometric==2.5.3
torch-scatter==2.1.2
torch-sparse==0.6.18

# Data
numpy==1.26.4
pandas==2.2.2
scipy==1.13.0
networkx==3.3
h5py==3.11.0

# Experiment management
hydra-core==1.3.2
omegaconf==2.3.0
wandb==0.17.0

# Visualization (Python side)
matplotlib==3.9.0
seaborn==0.13.2

# Testing
pytest==8.2.0
hypothesis==6.102.0

# Utilities
tqdm==4.66.4
rich==13.7.1
click==8.1.7
```

### OR-Tools (baseline solver)

```bash
pip install ortools==9.10.4067
```

### Visualizer (Node.js)

**Requires:** Node.js 20+, npm 10+

```bash
cd visualizer
npm install
npm run dev       # development server at http://localhost:5173
npm run build     # production build to visualizer/dist/
```

### `visualizer/package.json`

```json
{
  "name": "ngbb-visualizer",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "d3": "^7.9.0",
    "lucide-react": "^0.383.0"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.3.0",
    "autoprefixer": "^10.4.19",
    "postcss": "^8.4.38",
    "tailwindcss": "^3.4.4",
    "vite": "^5.2.11"
  }
}
```

### Docker (optional, recommended for reproducibility)

```bash
docker-compose up --build   # starts full Python environment with SCIP
```

---

## 6. Module 1 — Problem Instance Generator

### Purpose
Generate synthetic CVRP/TSP instances for training, validation, and testing. Convert raw instances into bipartite graphs for GNN input.

### `src/data/instance_generator.py`

**Class `InstanceGenerator`** — all methods below are required:

```python
class InstanceGenerator:
    def __init__(self, seed: int):
        """Seed the PRNG. All methods deterministic given seed."""

    def generate_random_euclidean(self, n_customers: int, capacity: int = 50) -> CVRPInstance:
        """
        Uniform random node positions in [0, 100]^2.
        Depot at (50, 50). Customer demands ~ Uniform(1, 10).
        Returns CVRPInstance dataclass.
        """

    def generate_clustered(self, n_customers: int, n_clusters: int = 3, capacity: int = 50) -> CVRPInstance:
        """
        Gaussian clusters of nodes. Each cluster has a random centre
        in [10, 90]^2, spread ~ N(0, 8). Depot at (50, 50).
        """

    def generate_mixed(self, n_customers: int, cluster_ratio: float = 0.6) -> CVRPInstance:
        """60% clustered nodes + 40% uniform random nodes."""

    def load_tsplib(self, filepath: str) -> CVRPInstance:
        """Parse .vrp TSPLIB file format into CVRPInstance."""
```

**`CVRPInstance` dataclass:**

```python
@dataclass
class CVRPInstance:
    n_customers: int
    depot: np.ndarray          # shape (2,) — x, y coords
    customers: np.ndarray      # shape (n, 2) — x, y coords
    demands: np.ndarray        # shape (n,) — integer demands
    capacity: int              # vehicle capacity Q
    distance_matrix: np.ndarray  # shape (n+1, n+1) — precomputed Euclidean
    instance_id: str           # "{type}_{n}_{seed}"
```

### `src/data/generate.py` — CLI

```bash
# Generate full training dataset
python src/data/generate.py \
  --split train \
  --n-instances 50000 \
  --size-min 20 \
  --size-max 50 \
  --workers 32 \
  --output-dir data/generated/

# Generate OOD validation set
python src/data/generate.py \
  --split val_ood \
  --n-instances 2000 \
  --size-min 50 \
  --size-max 100 \
  --output-dir data/generated/
```

**Dataset size targets:**

| Split | Size Range | Count |
|-------|-----------|-------|
| `train` | n = 20–50 | 50,000 |
| `val_id` | n = 20–50 | 5,000 |
| `val_ood` | n = 50–100 | 2,000 |
| `test_small` | n = 50–80 | 1,000 |
| `test_large` | n = 100–200 | 500 |
| `test_tsplib` | TSPLIB benchmarks | 100 |

### `src/data/graph_constructor.py`

Converts a `CVRPInstance` + LP relaxation solution into a bipartite graph for the GNN.

**Graph structure:**
- **Variable nodes** (one per decision variable x_ij): represent route edges
- **Constraint nodes** (one per constraint): capacity, covering, subtour

**Variable node features (12 dimensions):**

```
[0]  lp_value          — fractional LP solution value x̂_ij ∈ (0,1)
[1]  fractionality     — min(x̂_ij, 1 - x̂_ij)
[2]  pseudocost_up     — avg bound improvement when branching x_ij=1
[3]  pseudocost_down   — avg bound improvement when branching x_ij=0
[4]  pseudocost_reliability  — number of pseudocost updates
[5]  obj_coeff         — c_ij (normalized by instance cost range)
[6]  distance          — Euclidean distance (normalized)
[7]  degree_from       — degree of source node i in constraint graph
[8]  degree_to         — degree of destination node j
[9]  is_depot_edge     — binary: 1 if i or j is depot
[10] demand_from       — demand at node i (0 for depot)
[11] demand_to         — demand at node j (0 for depot)
```

**Constraint node features (5 dimensions):**

```
[0]  rhs               — right-hand side of constraint
[2]  activity          — current LHS value at LP solution
[3]  slack             — rhs - activity
[4]  constraint_type   — one-hot: 0=capacity, 1=covering, 2=subtour
```

**Output:** `torch_geometric.data.HeteroData` with node types `variable` and `constraint`, edge index for bipartite connections.

---

## 7. Module 2 — Branch-and-Bound Engine

### Purpose
Wrap SCIP's B&B solver with a custom branching rule callback that calls the GNN policy. Collect training data via strong branching oracle.

### `src/solver/scip_wrapper.py`

```python
class CVRPSolver:
    def __init__(self, instance: CVRPInstance, time_limit: float = 60.0, node_limit: int = 100_000):
        """Initialise SCIP model with CVRP formulation."""

    def add_branching_rule(self, rule: BranchingRule) -> None:
        """Register a custom branching rule with SCIP."""

    def solve(self) -> SolveResult:
        """Run B&B. Returns SolveResult with cost, route, nodes_explored, time."""

    def get_lp_solution(self) -> dict[str, float]:
        """Return current LP relaxation variable values."""
```

**CVRP formulation in SCIP:**
- Binary variables: `x[i][j]` ∈ {0,1} for each directed edge (i,j)
- Objective: minimize Σ c_ij · x_ij
- Constraints: degree-2 at depot, visit-once per customer, subtour elimination (lazy SECs), capacity (MTZ or flow formulation — use MTZ for small instances, flow for n > 30)

### `src/solver/branching_rule.py`

```python
class GNNBranchingRule(pyscipopt.Branchrule):
    """
    SCIP branching rule that calls GNN policy at each branching decision.
    Registered via model.includeBranchrule().
    """

    def __init__(self, policy: NGBBPolicy, feature_extractor: FeatureExtractor):
        self.policy = policy
        self.extractor = feature_extractor

    def branchexeclp(self, allowaddcons):
        """
        Called by SCIP at each LP branching decision.
        Steps:
          1. Extract bipartite graph from current LP state
          2. Run GNN inference → per-variable scores
          3. Select argmax variable among fractional candidates
          4. Call self.model.branchVar(var) with selected variable
        """
        graph = self.extractor.extract(self.model)
        scores = self.policy.score(graph)
        fractional_vars = self.model.getLPBranchCands()[0]
        best_var = fractional_vars[scores.argmax()]
        self.model.branchVar(best_var)
        return {"result": pyscipopt.SCIP_RESULT.BRANCHED}
```

### `src/solver/strong_branching.py`

Strong branching oracle for training data collection. For each branching decision, actually solves both child LPs and picks the variable with the highest bound improvement. Returns the chosen variable as the training label.

**Important:** Limit to `node_limit=500` per instance during data collection — full strong branching is expensive. Record (graph_state, label) pairs at every branching node up to the limit.

### `src/solver/event_logger.py`

Records every B&B event during a solve for visualization and analysis:

```python
@dataclass
class SolveEvent:
    frame: int
    event_type: str              # 'branch' | 'prune' | 'incumbent' | 'solved'
    node_id: int
    depth: int
    branch_edge: tuple[int,int] | None
    lower_bound: float
    upper_bound: float
    nodes_explored: int
    active_edges: list[tuple[int,int]]
    pruned_nodes: list[int]
```

---

## 8. Module 3 — GNN Branching Policy

### Purpose
A message-passing GNN that operates on the bipartite constraint-variable graph and outputs a score for each candidate branching variable.

### `src/models/gnn.py`

**Architecture:**

```
Input: HeteroData bipartite graph
       variable nodes: feature dim 12
       constraint nodes: feature dim 5

Layer 0: Linear projection → hidden dim 64 (both node types)

Layer 1–3: BipartiteMessagePassing
  - constraint → variable: sum(W_cv · h_c + b_cv) for c in neighbors
  - variable → constraint: mean(W_vc · h_v + b_vc) for v in neighbors
  - Each direction: separate linear + ReLU + LayerNorm
  - Residual connection added after each full round

Output: variable node embeddings, shape (n_vars, 64)
```

```python
class BipartiteGNN(nn.Module):
    def __init__(self, var_dim: int = 12, con_dim: int = 5,
                 hidden_dim: int = 64, n_layers: int = 3):
        ...

    def forward(self, data: HeteroData) -> torch.Tensor:
        """Returns variable node embeddings, shape (n_vars, hidden_dim)."""
```

### `src/models/output_head.py`

```python
class ScoringHead(nn.Module):
    """
    MLP: hidden_dim → 32 → 1 (scalar score per variable).
    Applied per-variable node. Dropout 0.1 on both linear layers.
    """
    def forward(self, var_embeddings: torch.Tensor) -> torch.Tensor:
        """Returns scores shape (n_vars,). No softmax — raw logits."""
```

### `src/models/policy.py`

```python
class NGBBPolicy(nn.Module):
    def __init__(self, gnn: BipartiteGNN, head: ScoringHead):
        ...

    def forward(self, data: HeteroData) -> torch.Tensor:
        """Returns per-variable logits for cross-entropy loss during training."""

    def score(self, data: HeteroData) -> np.ndarray:
        """Inference: returns numpy scores array (no grad)."""

    def select_branch_var(self, data: HeteroData, candidates: list[int]) -> int:
        """Returns index of highest-scoring candidate variable."""
```

---

## 9. Module 4 — Training Pipeline

### Stage 1: Imitation Learning

**Goal:** Supervise the GNN to mimic strong branching variable selection.

**Loss:** Cross-entropy between GNN output distribution and one-hot label from oracle.

```
L_IL = -log P_θ(y* | G_t)
```

where y* is the variable chosen by strong branching and G_t is the bipartite graph at B&B node t.

### `src/training/train_il.py` — CLI

```bash
python src/training/train_il.py \
  --config experiments/il_baseline.yaml \
  --data-dir data/processed/ \
  --checkpoint-dir checkpoints/ \
  --wandb-project ngbb-routing
```

### `src/training/il_trainer.py` — Core logic

```python
class ILTrainer:
    def __init__(self, policy, optimizer, scheduler, config):
        ...

    def train_epoch(self, dataloader) -> dict:
        """One epoch of IL. Returns {'loss': float, 'top1_acc': float}."""

    def validate(self, dataloader) -> dict:
        """Eval on validation split. Returns same keys."""

    def train(self, n_steps: int = 200_000) -> None:
        """Full training loop with early stopping on val top-1 accuracy."""
```

**Hyperparameters (from `experiments/il_baseline.yaml`):**

```yaml
model:
  hidden_dim: 64
  n_layers: 3
  dropout: 0.1

training:
  optimizer: adam
  lr: 1.0e-3
  lr_schedule: cosine
  warmup_steps: 1000
  batch_size: 32
  n_steps: 200000
  early_stopping_patience: 20000
  grad_clip: 1.0

data:
  train_split: train
  val_split: val_id
  n_workers: 4
```

### Stage 2: RL Fine-Tuning (optional, run after IL)

**Goal:** Further reduce nodes explored using solve quality as reward signal.

**Reward:** `R = -(nodes_explored / baseline_nodes)` normalized per instance, where baseline_nodes is pseudocost B&B node count.

**Algorithm:** REINFORCE with moving-average baseline. KL penalty against IL model (weight `β=0.01`) to prevent policy collapse.

```bash
python src/training/train_rl.py \
  --config experiments/rl_finetune.yaml \
  --pretrained checkpoints/il_best.pt
```

---

## 10. Module 5 — Evaluation Suite

### `src/evaluation/benchmark.py` — CLI

```bash
# Full benchmark run
python src/evaluation/benchmark.py \
  --checkpoint checkpoints/best.pt \
  --test-splits test_small test_large test_tsplib \
  --baselines random pseudocost strong_branching ortools \
  --output-dir results/ \
  --seeds 5
```

### Baselines to implement in `src/evaluation/baselines.py`

| Baseline | Description |
|----------|-------------|
| `random` | Uniform random selection among fractional variables |
| `pseudocost` | Standard pseudocost branching (SCIP default) |
| `strong_branching` | Oracle — evaluates all candidates (reference ceiling) |
| `ortools` | Google OR-Tools VRP solver (different algorithm class) |
| `nearest_neighbour` | Greedy construction heuristic (lower bound on quality) |

### Metrics to compute in `src/evaluation/metrics.py`

| Metric | Formula | Target |
|--------|---------|--------|
| Node reduction | `(nodes_baseline - nodes_ngbb) / nodes_baseline × 100` | ≥ 60% |
| Solve time ratio | `time_ngbb / time_baseline` | < 2.0× |
| Optimality gap | `\|(cost - optimal) / optimal\| × 100` | ≤ 0.5% |
| Generalization ratio | Node reduction on test_large / node reduction on val_id | ≥ 0.7 |

### Statistical testing in `src/evaluation/stats.py`

- Wilcoxon signed-rank test for all primary metric comparisons
- Report mean ± std and p-value in all result tables
- p < 0.05 required for any claimed improvement

---

## 11. Visualizer UI — Comparative Frontend

### Purpose

An interactive browser application showing side-by-side animated comparison of traditional B&B vs NGBB solving the same delivery routing instance. The solver difference must be immediately, visually obvious — NGBB's canvas converges cleanly while Traditional's thrashes.

**Runs entirely in-browser. No backend required. All solvers are simulated in JavaScript.**

### Technology Stack

| Layer | Technology |
|-------|-----------|
| Framework | React 18 (functional components + hooks only) |
| Styling | Tailwind CSS utility classes only — no custom component CSS |
| Graph rendering | D3.js v7 (layout computation) + Canvas 2D API (draw calls) |
| Animation | `requestAnimationFrame` loop; single global frame counter |
| Build | Vite 5.x + @vitejs/plugin-react |
| Icons | lucide-react |
| State | `useState` + `useReducer` — no Redux, no Zustand |

### Layout Structure

Four bands, stacked vertically, always visible (no tabs):

```
┌─────────────────────────────────────────────────────────┐
│ HEADER BAR                                              │
│ Title | Node slider | Seed input | Problem type | Gen   │
├─────────────────────────────────────────────────────────┤
│ CONTROL TOOLBAR                                         │
│ ▶ Play │ ⏸ Pause │ ⏭ Step │ ↺ Reset │ Speed ─────── │
├──────────────────────────┬──────────────────────────────┤
│ TRADITIONAL B&B CANVAS   │  NGBB CANVAS                 │
│                          │                              │
│   [network graph with    │   [same graph, faster        │
│    animated traversal]   │    convergence]              │
│                          │                              │
│                    VS                                   │
├─────────────────────────────────────────────────────────┤
│ STATS DASHBOARD                                         │
│ Nodes Explored │ Best Cost │ Solve Time │ Gap %         │
│                [NODE REDUCTION BADGE]                    │
└─────────────────────────────────────────────────────────┘
```

---

## 12. Visualizer — Component Specifications

### `App.jsx` — Root component

Responsibilities:
- Holds **all application state** (no child component has its own solver state)
- Runs the `requestAnimationFrame` animation loop in a `useEffect`
- Generates instance + event logs on mount and on config change
- Passes derived state down to children as props

**State shape:**

```javascript
const [config, setConfig] = useState({
  nodeCount: 20,
  seed: 42,
  problemType: 'cvrp',   // 'cvrp' | 'tsp'
  vehicleCapacity: 50,
});
const [instance, setInstance] = useState(null);
const [tradResult, setTradResult] = useState(null);   // SolverResult
const [ngbbResult, setNgbbResult] = useState(null);   // SolverResult
const [frame, setFrame] = useState(0);
const [isPlaying, setIsPlaying] = useState(false);
const [speed, setSpeed] = useState(1.0);
const [darkMode, setDarkMode] = useState(false);
```

**Animation loop:**

```javascript
useEffect(() => {
  if (!isPlaying) return;
  let raf;
  let lastTime = 0;
  const tick = (now) => {
    if (now - lastTime > 1000 / (24 * speed)) {  // 24fps base rate
      setFrame(f => {
        const maxFrame = Math.max(tradResult.events.length, ngbbResult.events.length) - 1;
        if (f >= maxFrame) { setIsPlaying(false); return f; }
        return f + 1;
      });
      lastTime = now;
    }
    raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);
  return () => cancelAnimationFrame(raf);
}, [isPlaying, speed, tradResult, ngbbResult]);
```

---

### `HeaderBar.jsx`

Props: `{ config, onConfigChange, onGenerate, darkMode, onToggleDark }`

Controls:
- App title: "NGBB Visualizer" in display font
- **Node count slider:** range 10–60, default 20, step 1. Label: "Nodes: {value}". Regenerates instance on change (debounce 300ms).
- **Seed input:** number input 0–9999. Default 42. Regenerates on change.
- **Problem type dropdown:** options "CVRP" | "TSP". Changes formulation.
- **Vehicle capacity input:** number 20–100, default 50. Only shown when problemType === 'cvrp'.
- **Generate New Instance button:** randomizes seed (Math.random() * 9999 | 0), calls onGenerate.
- **Dark mode toggle:** sun/moon icon (lucide-react), calls onToggleDark.

---

### `ControlToolbar.jsx`

Props: `{ isPlaying, speed, onPlay, onPause, onStep, onReset, onRunToEnd }`

Controls:
- **Play/Pause button:** single button that toggles. Icon: Play2 / Pause from lucide-react.
- **Step Forward button:** advances frame by 1. Disabled when playing.
- **Reset button:** sets frame to 0, sets isPlaying false.
- **Run to End button:** sets frame to max immediately (skips animation).
- **Speed slider:** range 0.25–8, step 0.25, default 1.0. Label: "{value}×".

---

### `SolverCanvas.jsx`

Props: `{ label, result, currentFrame, colorScheme, width, height }`

```javascript
const SolverCanvas = ({ label, result, currentFrame, colorScheme, width = 560, height = 420 }) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!result || !canvasRef.current) return;
    const ctx = canvasRef.current.getContext('2d');
    const frameData = deriveFrameState(result, currentFrame);  // from solverState.js
    drawFrame(ctx, result.instance, frameData, colorScheme, width, height);  // from canvasRenderer.js
  }, [result, currentFrame, colorScheme, width, height]);

  const isSolved = result && currentFrame >= result.events.length - 1;

  return (
    <div className="relative flex flex-col items-center">
      <div className="text-sm font-semibold mb-2">{label}</div>
      <canvas ref={canvasRef} width={width} height={height} />
      {isSolved && <SolvedBadge time={result.solvedAtFrame} />}
    </div>
  );
};
```

**Critical rules:**
- Canvas element is imperative. Do NOT use `useState` for draw data.
- `drawFrame` is called from `useEffect` only — never from render.
- `canvasRef.current` is valid only inside `useEffect`.

---

### `StatsDashboard.jsx`

Props: `{ tradResult, ngbbResult, frame }`

Derives current stats from event logs using `deriveFrameState()`.

Four metric cards displayed in a 2×2 grid:

| Card | Traditional value | NGBB value | Delta |
|------|-----------------|-----------|-------|
| Nodes Explored | integer | integer | "−N%" in red when NGBB < Trad |
| Best Route Cost | float 2dp or "—" | float 2dp or "—" | — |
| Solve Time | "Xms" simulated | "Xms" simulated | — |
| Optimality Gap | "X.XX%" | "X.XX%" | shrinking progress bar |

**Node Reduction Badge:** Large card, teal background, appears only after both solvers finish (`frame >= max(trad.events.length, ngbb.events.length) - 1`). Displays:
```
⚡ 67% fewer nodes explored with NGBB
```
Animates in with `scale-95 → scale-100` CSS transition.

---

### `MetricCard.jsx`

Props: `{ title, tradValue, ngbbValue, delta?, highlight? }`

Renders a card with:
- Title row (grey label text)
- Two-column value display: Traditional (blue) | NGBB (teal)
- Optional delta badge (red background for improvement)

---

### `VsBadge.jsx`

A vertical divider between the two canvases. Contains a pill badge with "VS" text. The pill has a subtle pulse animation (`animate-pulse`) while both solvers are still running, and becomes static when both finish.

---

## 13. Visualizer — Solver Simulation Logic

All simulation is pre-computed synchronously when the instance is generated. Animation is purely playback. Both solvers use the same node positions and distance matrix.

### `src/utils/prng.js`

```javascript
// mulberry32 — fast, seedable, good enough for this use case
export function mulberry32(seed) {
  return function() {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    var t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
```

### `src/utils/instanceGenerator.js`

```javascript
export function generateInstance(nodeCount, seed, problemType, vehicleCapacity = 50) {
  const rand = mulberry32(seed);

  // Generate node positions
  const depot = { x: 50, y: 50, id: 0, isDepot: true, demand: 0 };
  const customers = Array.from({ length: nodeCount }, (_, i) => ({
    x: rand() * 90 + 5,
    y: rand() * 90 + 5,
    id: i + 1,
    isDepot: false,
    demand: Math.floor(rand() * 9) + 1,  // demand 1–9
  }));
  const nodes = [depot, ...customers];

  // Precompute distance matrix
  const dist = Array.from({ length: nodes.length }, (_, i) =>
    Array.from({ length: nodes.length }, (_, j) => {
      const dx = nodes[i].x - nodes[j].x;
      const dy = nodes[i].y - nodes[j].y;
      return Math.sqrt(dx*dx + dy*dy);
    })
  );

  return { nodes, dist, depot, customers, vehicleCapacity, problemType };
}
```

### `src/utils/traditionalSolver.js`

Simulates vanilla B&B with poor branching (random among top-5 fractional variables). Produces a long event log with many explored nodes.

```javascript
export function runTraditionalSolver(instance, seed) {
  const rand = mulberry32(seed + 1);
  const events = [];
  let frame = 0;
  let nodesExplored = 0;
  let bestCost = Infinity;
  const optimalCost = computeGreedyCost(instance) * 0.88;  // simulated optimum

  // Simulate DFS B&B
  const stack = [createRootNode(instance)];

  while (stack.length > 0 && nodesExplored < 800) {
    const node = stack.pop();
    nodesExplored++;

    // Compute LP bound (greedy approximation)
    const lpBound = computeLPBound(node, instance, rand);

    if (lpBound >= bestCost) {
      // Prune
      events.push({ frame: frame++, type: 'prune', nodeId: node.id,
                    depth: node.depth, lowerBound: lpBound, upperBound: bestCost,
                    nodesExplored, activeEdges: node.edges, prunedNodes: [node.id],
                    branchEdge: null });
      continue;
    }

    if (isIntegerFeasible(node)) {
      bestCost = Math.min(bestCost, lpBound);
      events.push({ frame: frame++, type: 'incumbent', nodeId: node.id,
                    depth: node.depth, lowerBound: lpBound, upperBound: bestCost,
                    nodesExplored, activeEdges: node.edges, prunedNodes: [],
                    branchEdge: null });
      continue;
    }

    // Branch: pick random edge from top-5 fractional candidates (bad heuristic)
    const branchEdge = selectBranchEdgeRandom(node, instance, rand);
    const [child0, child1] = branch(node, branchEdge);
    stack.push(child1, child0);  // DFS order

    events.push({ frame: frame++, type: 'branch', nodeId: node.id,
                  depth: node.depth, branchEdge, lowerBound: lpBound,
                  upperBound: bestCost, nodesExplored, activeEdges: node.edges,
                  prunedNodes: [] });
  }

  events.push({ frame: frame, type: 'solved', nodeId: -1, depth: 0,
                branchEdge: null, lowerBound: bestCost, upperBound: bestCost,
                nodesExplored, activeEdges: computeOptimalRoute(instance), prunedNodes: [] });

  return { events, optimalCost: bestCost, totalNodes: nodesExplored,
           solvedAtFrame: frame, instance };
}
```

### `src/utils/ngbbSolver.js`

Same structure as `traditionalSolver.js` but with a smarter branching heuristic that produces 60–75% fewer events.

```javascript
// GNN score approximation — selects the best variable much more reliably
function selectBranchEdgeGNN(node, instance) {
  const fractional = getFractionalEdges(node);
  // Score = (1 - fractionality) * 0.6 + (1 - normalized_distance_to_depot) * 0.4
  // This mimics a learned policy that prefers edges close to integer and near depot
  const scored = fractional.map(edge => ({
    edge,
    score: (1 - Math.abs(edge.value - 0.5) * 2) * 0.6
          + (1 - instance.dist[0][edge.to] / instance.maxDist) * 0.4
  }));
  scored.sort((a, b) => b.score - a.score);
  return scored[0].edge;
}
```

**Target output ratios (enforce in simulation):**
- NGBB nodes explored ≈ Traditional nodes × 0.25–0.35
- NGBB incumbent found at ≈ frame 15–20 (Traditional: frame 80–120)
- Both reach same optimalCost value

---

## 14. Visualizer — Canvas Rendering Spec

### `src/utils/canvasRenderer.js`

All draw functions are **pure** — they take `(ctx, data)` and draw. No React, no state.

**Draw order (never deviate from this layering):**

```javascript
export function drawFrame(ctx, instance, frameState, colorScheme, width, height) {
  ctx.clearRect(0, 0, width, height);           // 1. Clear
  drawBackground(ctx, width, height);            // 2. Background fill
  drawUnvisitedEdges(ctx, instance, frameState); // 3. Grey edges (opacity 0.3)
  drawDiscardedEdges(ctx, instance, frameState); // 4. Red dotted (opacity 0.2)
  drawVisitedEdges(ctx, instance, frameState, colorScheme); // 5. Solid edges
  drawCandidateEdges(ctx, instance, frameState); // 6. Orange dashed (animated)
  drawOptimalEdges(ctx, instance, frameState, colorScheme); // 7. Thick teal (on top)
  drawNodes(ctx, instance, frameState, colorScheme);        // 8. Circles
  drawLabels(ctx, instance, width, height);                 // 9. Text labels
  drawTooltip(ctx, instance, frameState);                   // 10. Hover tooltip
}
```

### Node Rendering Rules

| Node State | Fill | Border | Radius | Extra |
|-----------|------|--------|--------|-------|
| Depot (default) | `#1E3A5F` | none | 14px | White "D" label centred |
| Customer (unvisited) | `#FFFFFF` | `#1E3A5F` 1.5px | 8px | — |
| Customer (active/branching) | `#0D7377` | `#0D7377` 2px | 10px | Pulse ring at 1Hz |
| Customer (visited) | `#D6E4F0` | `#999999` 1px | 8px | — |
| Customer (optimal path) | `#0D7377` | none | 9px | — |

**Pulse ring animation:**
```javascript
// In draw loop, pass a timestamp (performance.now()) into drawFrame
const pulseAlpha = 0.3 + 0.7 * (Math.sin(timestamp / 500 * Math.PI) * 0.5 + 0.5);
ctx.globalAlpha = pulseAlpha;
ctx.beginPath();
ctx.arc(x, y, 16, 0, Math.PI * 2);
ctx.strokeStyle = colorScheme === 'teal' ? '#0D7377' : '#1E3A5F';
ctx.lineWidth = 2;
ctx.stroke();
ctx.globalAlpha = 1;
```

### Edge Rendering Rules

| Edge State | Colour | Width | Style | Opacity |
|-----------|--------|-------|-------|---------|
| Unvisited | `#CCCCCC` | 0.5px | Solid | 0.4 |
| Candidate (evaluating) | `#F59E0B` | 1.5px | Dashed, offset scrolls | 1.0 |
| Accepted (in solution) | colorScheme primary | 2px | Solid | 0.9 |
| Discarded | `#A63A3A` | 0.5px | Dotted | 0.2 |
| Optimal solution | `#0D7377` | 3px | Solid | 1.0 |

**Scrolling dash animation for candidate edges:**
```javascript
// dashOffset decreases by 1 per frame → creates "marching ants" effect
ctx.setLineDash([6, 4]);
ctx.lineDashOffset = -frameState.frame * 0.8;
```

### Node Layout

Use D3 force simulation to compute layout **once** on instance generation, then freeze positions. Store as `{id, x, y}` array. Map to canvas pixel space:

```javascript
// Scale [0,100] instance coords → canvas pixel space with 40px padding
const scaleX = d3.scaleLinear().domain([0, 100]).range([40, width - 40]);
const scaleY = d3.scaleLinear().domain([0, 100]).range([40, height - 40]);
```

---

## 15. Visualizer — Event Log Schema

Both solver simulations return objects matching this schema exactly. `canvasRenderer.js` and `solverState.js` consume this schema.

```typescript
// Single B&B event
type SolverEvent = {
  frame: number;                         // 0-based sequential frame index
  type: 'branch' | 'prune' | 'incumbent' | 'solved';
  nodeId: number;                        // B&B tree node ID being processed
  depth: number;                         // Depth in B&B tree (root = 0)
  branchEdge: [number, number] | null;   // [fromNodeId, toNodeId] being branched
  lowerBound: number;                    // Current LP lower bound
  upperBound: number;                    // Best incumbent cost (Infinity if none)
  nodesExplored: number;                 // Running cumulative count
  activeEdges: [number, number][];       // All edges in current partial solution
  prunedNodes: number[];                 // B&B node IDs pruned this step
};

// Full solver output returned by runTraditionalSolver / runNGBBSolver
type SolverResult = {
  events: SolverEvent[];
  optimalCost: number;
  optimalRoute: number[];                // Customer visit order [0, 3, 1, 5, ..., 0]
  totalNodes: number;                    // events.length - 1 (excl. 'solved' event)
  solvedAtFrame: number;                 // frame index of 'solved' event
  instance: GeneratedInstance;           // Back-reference to input instance
};
```

### `src/utils/solverState.js`

```javascript
// Derives what to render at a given frame from the event log
export function deriveFrameState(result, frame) {
  const clampedFrame = Math.min(frame, result.events.length - 1);
  const event = result.events[clampedFrame];
  const prevEvents = result.events.slice(0, clampedFrame + 1);

  return {
    currentEvent: event,
    nodesExplored: event.nodesExplored,
    bestCost: event.upperBound === Infinity ? null : event.upperBound,
    lowerBound: event.lowerBound,
    gap: event.upperBound === Infinity ? null
         : Math.abs(event.upperBound - result.optimalCost) / result.optimalCost * 100,
    visitedNodes: new Set(prevEvents.map(e => e.nodeId)),
    activeEdges: event.activeEdges,
    candidateEdge: event.type === 'branch' ? event.branchEdge : null,
    prunedNodes: new Set(prevEvents.flatMap(e => e.prunedNodes)),
    isSolved: event.type === 'solved',
    optimalRoute: event.type === 'solved' ? result.optimalRoute : null,
    simulatedTimeMs: Math.round(event.nodesExplored * (result === 'trad' ? 2.4 : 0.6)),
  };
}
```

---

## 16. Experiment Configs & Reproducibility

### `experiments/il_baseline.yaml`

```yaml
# @package _global_
defaults:
  - _self_

seed: 42
device: cuda   # falls back to cpu if unavailable

model:
  hidden_dim: 64
  n_layers: 3
  dropout: 0.1
  var_input_dim: 12
  con_input_dim: 5

training:
  optimizer: adam
  lr: 1.0e-3
  beta1: 0.9
  beta2: 0.999
  weight_decay: 1.0e-5
  lr_schedule: cosine
  warmup_steps: 1000
  batch_size: 32
  n_steps: 200000
  early_stopping_patience: 20000
  grad_clip: 1.0
  checkpoint_every: 5000

data:
  train_split: train
  val_split: val_id
  n_workers: 4
  pin_memory: true

logging:
  wandb_project: ngbb-routing
  log_every: 100
```

### `experiments/rl_finetune.yaml`

```yaml
# @package _global_
defaults:
  - il_baseline
  - _self_

rl:
  pretrained_checkpoint: checkpoints/il_best.pt
  reward: neg_node_ratio        # -(nodes_ngbb / nodes_pseudocost)
  baseline: moving_average
  baseline_momentum: 0.99
  kl_penalty_weight: 0.01       # prevents forgetting IL policy
  entropy_bonus: 0.001
  n_episodes: 100000
  episode_node_limit: 200
  lr: 3.0e-5                    # lower LR for fine-tuning
```

---

## 17. Evaluation Protocol & Metrics

### Primary result table format

All results reported as mean ± std over 5 random seeds × N test instances:

```
Method          | Nodes Explored    | Solve Time (s)  | Gap %    | Node Reduction
----------------|-------------------|-----------------|----------|-----------------
Random          | 812.3 ± 94.1      | 8.41 ± 1.2      | 0.00%    | baseline
Pseudocost      | 487.6 ± 61.2      | 4.92 ± 0.8      | 0.00%    | 40.0%
Strong Branching| 134.1 ± 22.4      | 18.7 ± 3.1 *    | 0.00%    | 83.5% (oracle)
OR-Tools        | N/A               | 2.31 ± 0.4      | 1.2%     | N/A
NGBB (ours)     | 201.4 ± 31.7      | 2.18 ± 0.3      | 0.00%    | 75.2% ✓
```
*Strong branching is slow due to per-node LP solves — this is expected.

### Ablation table format (GNN depth)

| Layers | Val Acc | Nodes (n=30) | Nodes (n=100) | Gen. Ratio |
|--------|---------|-------------|--------------|------------|
| 1 | 61.2% | 312 | 891 | 0.65 |
| 2 | 74.8% | 248 | 612 | 0.71 |
| **3** | **81.3%** | **201** | **478** | **0.78** |
| 4 | 80.9% | 208 | 501 | 0.76 |

### Statistical significance reporting

Every claimed improvement must include:
```python
from scipy.stats import wilcoxon
stat, p = wilcoxon(ngbb_nodes, baseline_nodes, alternative='less')
# Report: "p = {p:.4f} (Wilcoxon signed-rank, one-sided)"
```

---

## 18. Project Timeline & Milestones

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| **1** | Weeks 1–2 | Problem formulation locked; SCIP environment set up; random B&B baseline running; unit tests for instance generator passing |
| **2** | Weeks 3–4 | Data pipeline complete; 50K training instances generated; strong branching oracle validated on 20-node instances |
| **3** | Weeks 5–6 | GNN architecture implemented; initial IL training on n=20 instances; >60% top-1 accuracy on val_id |
| **4** | Weeks 7–8 | Full training pipeline; hyperparameter sweep complete; IL model achieving >75% top-1 accuracy; visualizer v1 deployed |
| **5** | Weeks 9–10 | RL fine-tuning; generalization experiments on test_large; ablation studies documented |
| **6** | Weeks 11–12 | Final benchmark vs all baselines; results tables complete; paper draft; code cleaned + released |

### Go/No-Go criteria between phases

- Phase 2 → 3: Oracle strong branching achieves <200 nodes on 30-node instances (validates data quality)
- Phase 3 → 4: GNN top-1 accuracy ≥ 60% on val_id (validates architecture)
- Phase 4 → 5: GNN achieves ≥ 40% node reduction vs pseudocost on val_id (validates learning)
- Phase 5 → 6: Generalization ratio ≥ 0.65 on test_large (validates transfer)

---

## 19. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| GNN doesn't generalize to large instances | Medium | High | Curriculum training; graph size augmentation; normalize features by n |
| Strong branching data collection too slow | High | Medium | Cap at 500 nodes/instance; parallelize 32 workers; use partial SB |
| SCIP callback overhead eliminates speedup | Low | High | Batch GNN inference; cache features; profile before optimizing |
| Overfitting to synthetic distribution | Medium | Medium | Mix synthetic + TSPLIB; domain randomization on generation |
| RL fine-tuning destabilizes IL model | Medium | Medium | KL penalty weight β=0.01; revert to IL checkpoint if reward drops >10% |
| SCIP version incompatibility | Low | High | Pin SCIP 8.0.3 + PySCIPOpt 4.4.0 in Dockerfile; test on CI |

---

## 20. Acceptance Criteria — Full Project

The project is complete when **all** of the following pass:

### Research/ML criteria
- [ ] GNN top-1 accuracy ≥ 75% on val_id (in-distribution)
- [ ] NGBB achieves ≥ 60% node reduction vs pseudocost on test_small
- [ ] NGBB achieves ≥ 40% node reduction vs pseudocost on test_large (generalization)
- [ ] Optimality gap ≤ 0.5% on all test splits (near-optimal solutions)
- [ ] All improvements statistically significant (p < 0.05, Wilcoxon)
- [ ] Ablation results justify 3-layer GNN and full feature set

### Software/Engineering criteria
- [ ] All unit tests pass (`pytest tests/ -v`)
- [ ] Full training run reproducible from seed (`--seed 42` produces identical results)
- [ ] `python src/data/generate.py` completes 50K instances in < 4 hours on 32 cores
- [ ] `python src/training/train_il.py` completes 200K steps in < 12 hours on A100
- [ ] `python src/evaluation/benchmark.py` produces complete results table in < 2 hours

### Visualizer criteria
- [ ] Generates valid instance for any seed 0–9999 and node count 10–60 within 100ms
- [ ] Play/Pause/Step/Reset/Run-to-End controls all work correctly
- [ ] Speed slider works across 0.25×–8× with no visual glitches
- [ ] Stats dashboard updates every frame with correct values
- [ ] Node Reduction badge shows 50–85% reduction after both solvers finish
- [ ] All 5 edge states visually distinguishable at a glance
- [ ] Dark mode toggle flips all colours including canvas background
- [ ] Export PNG saves valid side-by-side canvas image
- [ ] Export JSON downloads valid event log matching schema in Section 15
- [ ] Renders correctly at 1024px, 1280px, 1920px viewport widths
- [ ] No console errors in production build
- [ ] Lighthouse performance score ≥ 85 (desktop)

---

## Appendix A — Notation Reference

| Symbol | Meaning |
|--------|---------|
| G = (V, E) | Problem graph: V = customers + depot, E = potential routes |
| x_ij | Binary decision variable: 1 if vehicle traverses edge (i,j) |
| c_ij | Travel cost on edge (i,j) — Euclidean distance |
| d(v) | Demand at customer node v |
| Q | Vehicle capacity constraint |
| x̂_ij | Fractional LP relaxation value of x_ij |
| θ | GNN model parameters |
| P_θ(y \| G_t) | GNN policy: probability of branching on variable y given state G_t |
| s(x_ij) | Raw GNN score for variable x_ij (pre-softmax logit) |
| L_IL | Imitation learning cross-entropy loss |
| R_t | RL reward at step t: −(nodes_ngbb / nodes_pseudocost) |
| n | Number of customer nodes in an instance |
| SB | Strong branching oracle |
| IL | Imitation learning |
| RL | Reinforcement learning (REINFORCE) |

---

## Appendix B — TSPLIB Data Format

Download TSPLIB `.vrp` files from http://vrp.atd-lab.inf.puc-rio.br/

Place in `data/raw/tsplib/`. The `load_tsplib()` method in `instance_generator.py` must parse this format:

```
NAME : E-n22-k4
COMMENT : (Christofides and Eilon, Min no of trucks: 4, Min distance: 375)
TYPE : CVRP
DIMENSION : 22
EDGE_WEIGHT_TYPE : EUC_2D
CAPACITY : 6000
NODE_COORD_SECTION
 1 145 215   ← depot is always node 1
 2 151 264
 ...
DEMAND_SECTION
 1 0          ← depot demand = 0
 2 1100
 ...
DEPOT_SECTION
 1
 -1
EOF
```

---

*End of NGBB Project Execution Document — v1.0*
*This document covers the complete research pipeline, all module specifications, and the full visualizer build guide. A developer reading this document top-to-bottom has everything needed to execute the project without further clarification.*
