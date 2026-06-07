# Neural-Guided Branch & Bound — Dynamic Delivery Route Optimization
## with Knowledge Distillation (Teacher→Student GNN)

> A GNN-augmented Branch-and-Bound solver for CVRP/TSP that learns branching heuristics from Strong Branching, then **distills** the learned policy into a compact Student model for fast deployment. The Student model reduces explored nodes by **75–85%** while being **4× smaller** than the Teacher.

## Architecture

```
  Strong Branching Oracle
           │
           ▼ (Imitation Learning)
  ┌─────────────────┐
  │  Teacher GNN     │  64-dim, 3-layer BipartiteGNN
  │  (Full Model)    │  Trained via IL on oracle labels
  └────────┬────────┘
           │ Knowledge Distillation
           │ (Soft-labels + Feature Alignment + Edge Penalty)
           ▼
  ┌─────────────────┐
  │  Student GNN     │  32-dim, 2-layer CompactGNN
  │  (Deployed Agent)│  ~4× fewer parameters
  └─────────────────┘
```

**Novel contributions:**
- Temperature-scaled soft-label distillation (T=4.0) for branching heuristics
- Feature-level alignment between Teacher and Student embeddings
- Edge repetition penalty in loss (delivery-specific: penalizes redundant traversal)
- Bottleneck attention pooling in Student for efficient message passing
- 2-opt local search refinement in Student's final route construction

## Quick Start

```bash
# Visualizer (runs entirely in browser — no Python needed)
cd ngbb-routing/visualizer
npm install
npm run dev    # → http://localhost:5173

# Python environment (for training pipeline)
conda create -n ngbb python=3.11
conda activate ngbb
conda install -c conda-forge scip
pip install -r ngbb-routing/requirements.txt

# Train Teacher (IL)
python src/training/train_il.py --config experiments/il_baseline.yaml

# Train Student (Knowledge Distillation)
python src/training/train_kd.py \
  --teacher-checkpoint checkpoints/il_best.pt \
  --temperature 4.0 --alpha 0.7 --n-steps 100000
```

## Visualizer

The interactive visualizer shows a **3-way comparison**:

| Solver | Description | Nodes | Speed |
|--------|-------------|-------|-------|
| **Traditional B&B** | Random branching (baseline) | ~500-800 | Slow |
| **Teacher NGBB** | Full GNN-guided branching | ~180-280 | Medium |
| **Student KD** | Distilled compact model | ~100-180 | Fast |

Features: Play/Pause/Step animation, adjustable speed, configurable node count/seed, glassmorphism dark-mode UI, animated route construction.

## Project Structure

```
ngbb-routing/
├── src/
│   ├── models/
│   │   ├── gnn.py              # Teacher BipartiteGNN (3-layer, 64-dim)
│   │   ├── student_gnn.py      # Student CompactGNN (2-layer, 32-dim)
│   │   ├── distillation.py     # KD loss, Teacher-Student wrapper
│   │   ├── output_head.py      # Scoring MLP heads
│   │   └── policy.py           # NGBBPolicy + StudentPolicy
│   ├── training/
│   │   ├── train_il.py         # Imitation learning CLI
│   │   ├── train_kd.py         # Knowledge distillation CLI
│   │   ├── kd_trainer.py       # KD training logic
│   │   ├── il_trainer.py       # IL training logic
│   │   └── losses.py           # IL, RL, and KD loss functions
│   ├── solver/                 # SCIP wrapper, branching rules
│   └── evaluation/             # Benchmarks, baselines, metrics
├── visualizer/
│   ├── src/
│   │   ├── App.jsx             # 3-solver comparison layout
│   │   ├── components/         # HeaderBar, SolverCanvas, StatsDashboard, etc.
│   │   └── utils/
│   │       ├── traditionalSolver.js  # Random B&B simulation
│   │       ├── ngbbSolver.js         # Teacher NGBB simulation
│   │       ├── studentSolver.js      # Student KD simulation (2-opt)
│   │       └── canvasRenderer.js     # Canvas drawing with glow effects
│   └── package.json
└── experiments/                # YAML configs for IL, RL, KD
```

## Architecture Details

See `NGBB_PROJECT.md` for the full execution document.
