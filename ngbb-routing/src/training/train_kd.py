"""CLI: Knowledge Distillation training — Teacher GNN -> Student GNN.

Usage:
    python src/training/train_kd.py \
      --teacher-checkpoint checkpoints/il_best.pt \
      --data-dir data/processed/ \
      --checkpoint-dir checkpoints/ \
      --temperature 4.0 --alpha 0.7 --n-steps 100000
"""

import argparse
import os
import yaml
import torch
from pathlib import Path

from src.models.policy import NGBBPolicy
from src.models.distillation import StudentPolicy
from src.training.kd_trainer import KDTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation: Teacher -> Student GNN")
    parser.add_argument("--teacher-checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default="data/processed/")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/")
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--n-steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Load config
    config = {
        "device": device,
        "distillation": {
            "temperature": args.temperature,
            "alpha": args.alpha,
            "feature_weight": 0.1,
            "edge_penalty_weight": 0.05,
        },
        "training": {
            "lr": args.lr,
            "weight_decay": 1e-4,
            "warmup_steps": 500,
            "n_steps": args.n_steps,
            "grad_clip": 1.0,
            "batch_size": args.batch_size,
        },
    }

    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            file_config = yaml.safe_load(f)
        config.update(file_config)

    # Load Teacher
    print(f"Loading Teacher from {args.teacher_checkpoint}...")
    teacher = NGBBPolicy.build(var_dim=12, con_dim=5, hidden_dim=64, n_layers=3)
    ckpt = torch.load(args.teacher_checkpoint, map_location=device)
    teacher.load_state_dict(ckpt["model_state_dict"])
    teacher.eval()

    # Create Student
    student = StudentPolicy.build(var_dim=12, con_dim=5, hidden_dim=32, n_layers=2)

    # Compare sizes
    t_params = sum(p.numel() for p in teacher.parameters())
    s_params = sum(p.numel() for p in student.parameters())
    print(f"Teacher: {t_params:,} params | Student: {s_params:,} params")
    print(f"Compression: {t_params/s_params:.1f}x smaller")

    # Initialize trainer
    trainer = KDTrainer(teacher, student, config)

    print(f"\nStarting KD training for {args.n_steps} steps...")
    print(f"Temperature={args.temperature}, Alpha={args.alpha}")
    print(f"Device: {device}\n")

    # Save checkpoint dir
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("Knowledge Distillation training pipeline ready.")
    print("Connect a DataLoader with processed bipartite graph data to begin.")


if __name__ == "__main__":
    main()
