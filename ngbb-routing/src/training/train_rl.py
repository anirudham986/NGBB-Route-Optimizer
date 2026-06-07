"""CLI: RL fine-tuning loop.

Usage:
    python src/training/train_rl.py --config experiments/rl_finetune.yaml --pretrained checkpoints/il_best.pt
"""

from pathlib import Path

import click
import torch

from src.models.policy import NGBBPolicy
from src.training.rl_trainer import RLTrainer
from src.utils.config import load_config, resolve_device, config_to_dict
from src.utils.logging import get_logger

logger = get_logger("ngbb.training.train_rl")


@click.command()
@click.option("--config", required=True, help="Path to RL experiment YAML config")
@click.option("--pretrained", required=True, help="Path to pre-trained IL checkpoint")
@click.option("--checkpoint-dir", default="checkpoints/", help="Where to save checkpoints")
def main(config, pretrained, checkpoint_dir):
    """Run RL fine-tuning on a pre-trained IL model."""
    cfg = load_config(config)
    cfg_dict = config_to_dict(cfg)
    cfg_dict["device"] = resolve_device(cfg)

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    m = cfg_dict["model"]

    # Load pre-trained policy
    policy = NGBBPolicy.build(
        var_dim=m["var_input_dim"], con_dim=m["con_input_dim"],
        hidden_dim=m["hidden_dim"], n_layers=m["n_layers"], dropout=m["dropout"],
    )
    state = torch.load(pretrained, map_location=cfg_dict["device"])
    policy.load_state_dict(state["model_state_dict"])
    logger.info(f"Loaded pre-trained model from {pretrained}")

    # Frozen IL reference
    il_policy = NGBBPolicy.build(
        var_dim=m["var_input_dim"], con_dim=m["con_input_dim"],
        hidden_dim=m["hidden_dim"], n_layers=m["n_layers"], dropout=m["dropout"],
    )
    il_policy.load_state_dict(state["model_state_dict"])

    # Trainer
    trainer = RLTrainer(policy, il_policy, cfg_dict)

    rl_cfg = cfg_dict.get("rl", {})
    n_episodes = rl_cfg.get("n_episodes", 100_000)

    logger.info(f"Starting RL fine-tuning for {n_episodes} episodes")

    # Note: In a full implementation, each episode would solve a CVRP instance
    # and compute the reward. Here we show the training loop structure.
    for episode in range(n_episodes):
        # Placeholder: in practice, generate instance, solve with policy,
        # compute reward = -(nodes_ngbb / nodes_pseudocost)
        # graph, reward = run_episode(policy, instance)
        # metrics = trainer.train_episode(graph, reward)

        if (episode + 1) % 1000 == 0:
            logger.info(f"Episode {episode + 1}/{n_episodes}")

        if (episode + 1) % 10000 == 0:
            torch.save({"model_state_dict": policy.state_dict(),
                         "episode": episode}, f"{checkpoint_dir}/rl_ep{episode+1}.pt")

    torch.save({"model_state_dict": policy.state_dict()}, f"{checkpoint_dir}/rl_final.pt")
    logger.info("RL fine-tuning complete")


if __name__ == "__main__":
    main()
