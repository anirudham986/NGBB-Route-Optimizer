"""CLI: Imitation learning training loop.

Usage:
    python src/training/train_il.py --config experiments/il_baseline.yaml
"""

from pathlib import Path

import click
import torch
from torch_geometric.loader import DataLoader

from src.models.policy import NGBBPolicy
from src.data.dataset import NGBBDataset
from src.training.il_trainer import ILTrainer
from src.utils.config import load_config, resolve_device, config_to_dict
from src.utils.logging import get_logger

logger = get_logger("ngbb.training.train_il")


@click.command()
@click.option("--config", required=True, type=str, help="Path to experiment YAML config")
@click.option("--data-dir", default="data/processed/", help="Directory with processed .pt files")
@click.option("--checkpoint-dir", default="checkpoints/", help="Where to save checkpoints")
@click.option("--wandb-project", default=None, help="W&B project name (None to disable)")
@click.option("--resume", default=None, help="Path to checkpoint to resume from")
def main(config, data_dir, checkpoint_dir, wandb_project, resume):
    """Run imitation learning training."""
    cfg = load_config(config)
    cfg_dict = config_to_dict(cfg)
    cfg_dict["device"] = resolve_device(cfg)

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Config loaded: {config}")
    logger.info(f"Device: {cfg_dict['device']}")

    # Init wandb if requested
    if wandb_project:
        import wandb
        wandb.init(project=wandb_project, config=cfg_dict)

    # Build model
    m = cfg_dict["model"]
    policy = NGBBPolicy.build(
        var_dim=m["var_input_dim"], con_dim=m["con_input_dim"],
        hidden_dim=m["hidden_dim"], n_layers=m["n_layers"], dropout=m["dropout"],
    )
    logger.info(f"Model params: {sum(p.numel() for p in policy.parameters()):,}")

    # Load data
    train_ds = NGBBDataset(root=f"{data_dir}/{cfg_dict['data']['train_split']}")
    val_ds = NGBBDataset(root=f"{data_dir}/{cfg_dict['data']['val_split']}")
    t = cfg_dict["training"]
    train_loader = DataLoader(train_ds, batch_size=t["batch_size"],
                               shuffle=True, num_workers=t.get("n_workers", 4))
    val_loader = DataLoader(val_ds, batch_size=t["batch_size"],
                             num_workers=t.get("n_workers", 4))

    # Trainer
    trainer = ILTrainer(policy, cfg_dict)
    if resume:
        trainer.load_checkpoint(resume)
        logger.info(f"Resumed from {resume} at step {trainer.global_step}")

    # Training loop
    patience = t["early_stopping_patience"]
    best_acc = trainer.best_val_acc

    for epoch in range(t["n_steps"] // max(len(train_loader), 1) + 1):
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.validate(val_loader)

        logger.info(
            f"Epoch {epoch} | Step {trainer.global_step} | "
            f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['top1_acc']:.3f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['top1_acc']:.3f}"
        )

        if wandb_project:
            import wandb
            wandb.log({**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()},
                       "step": trainer.global_step})

        # Checkpointing
        if val_metrics["top1_acc"] > best_acc:
            best_acc = val_metrics["top1_acc"]
            trainer.best_val_acc = best_acc
            trainer.save_checkpoint(f"{checkpoint_dir}/il_best.pt")
            trainer.patience_counter = 0
            logger.info(f"New best val acc: {best_acc:.4f}")
        else:
            trainer.patience_counter += len(train_loader)
            if trainer.patience_counter >= patience:
                logger.info("Early stopping triggered")
                break

        if trainer.global_step >= t["n_steps"]:
            break

    trainer.save_checkpoint(f"{checkpoint_dir}/il_final.pt")
    logger.info("Training complete")


if __name__ == "__main__":
    main()
