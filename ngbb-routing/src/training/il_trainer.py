"""Core imitation learning training logic.

Trains the GNN policy to mimic strong branching variable selection
using cross-entropy loss with cosine LR schedule and early stopping.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch_geometric.loader import DataLoader

from src.models.policy import NGBBPolicy
from src.training.losses import il_cross_entropy_loss, compute_top_k_accuracy
from src.utils.logging import get_logger

logger = get_logger("ngbb.training.il")


class ILTrainer:
    """Imitation learning trainer for the NGBB policy."""

    def __init__(self, policy: NGBBPolicy, config: dict):
        self.policy = policy
        self.config = config
        self.device = config.get("device", "cpu")

        self.policy.to(self.device)

        self.optimizer = Adam(
            self.policy.parameters(),
            lr=config["training"]["lr"],
            betas=(config["training"]["beta1"], config["training"]["beta2"]),
            weight_decay=config["training"]["weight_decay"],
        )

        # LR schedule: warmup + cosine
        warmup = LinearLR(self.optimizer, start_factor=0.01, total_iters=config["training"]["warmup_steps"])
        cosine = CosineAnnealingLR(self.optimizer, T_max=config["training"]["n_steps"])
        self.scheduler = SequentialLR(self.optimizer, [warmup, cosine],
                                      milestones=[config["training"]["warmup_steps"]])

        self.grad_clip = config["training"]["grad_clip"]
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.global_step = 0

    def train_epoch(self, dataloader: DataLoader) -> dict:
        """One epoch of IL training.

        Returns:
            Dict with 'loss' and 'top1_acc'.
        """
        self.policy.train()
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        for batch in dataloader:
            batch = batch.to(self.device)
            labels = batch.y if hasattr(batch, 'y') else batch["label"]

            logits = self.policy(batch)
            loss = il_cross_entropy_loss(logits.unsqueeze(0) if logits.dim() == 1 else logits,
                                         labels)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            total_loss += loss.item()
            total_acc += compute_top_k_accuracy(
                logits.unsqueeze(0) if logits.dim() == 1 else logits, labels
            )
            n_batches += 1

        return {
            "loss": total_loss / max(n_batches, 1),
            "top1_acc": total_acc / max(n_batches, 1),
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict:
        """Evaluate on validation split.

        Returns:
            Dict with 'loss' and 'top1_acc'.
        """
        self.policy.eval()
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        for batch in dataloader:
            batch = batch.to(self.device)
            labels = batch.y if hasattr(batch, 'y') else batch["label"]
            logits = self.policy(batch)
            loss = il_cross_entropy_loss(
                logits.unsqueeze(0) if logits.dim() == 1 else logits, labels
            )
            total_loss += loss.item()
            total_acc += compute_top_k_accuracy(
                logits.unsqueeze(0) if logits.dim() == 1 else logits, labels
            )
            n_batches += 1

        return {
            "loss": total_loss / max(n_batches, 1),
            "top1_acc": total_acc / max(n_batches, 1),
        }

    def save_checkpoint(self, filepath: str, extra: dict | None = None):
        """Save model checkpoint."""
        state = {
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "best_val_acc": self.best_val_acc,
            "config": self.config,
        }
        if extra:
            state.update(extra)
        torch.save(state, filepath)

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        state = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.global_step = state.get("global_step", 0)
        self.best_val_acc = state.get("best_val_acc", 0.0)
