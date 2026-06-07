"""Hydra config resolution helpers.

Provides utilities for loading and merging experiment configs
with sensible defaults and environment variable overrides.
"""

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


# Default config values used when keys are missing from experiment YAML
DEFAULTS = {
    "seed": 42,
    "device": "cuda",
    "model": {
        "hidden_dim": 64,
        "n_layers": 3,
        "dropout": 0.1,
        "var_input_dim": 12,
        "con_input_dim": 5,
    },
    "training": {
        "optimizer": "adam",
        "lr": 1e-3,
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 1e-5,
        "lr_schedule": "cosine",
        "warmup_steps": 1000,
        "batch_size": 32,
        "n_steps": 200_000,
        "early_stopping_patience": 20_000,
        "grad_clip": 1.0,
        "checkpoint_every": 5000,
    },
    "data": {
        "train_split": "train",
        "val_split": "val_id",
        "n_workers": 4,
        "pin_memory": True,
    },
    "logging": {
        "wandb_project": "ngbb-routing",
        "log_every": 100,
    },
}


def load_config(config_path: str | Path) -> DictConfig:
    """Load a YAML config and merge with defaults.

    Args:
        config_path: Path to the experiment YAML file.

    Returns:
        Merged OmegaConf DictConfig with all defaults filled.
    """
    defaults_cfg = OmegaConf.create(DEFAULTS)
    file_cfg = OmegaConf.load(str(config_path))
    merged = OmegaConf.merge(defaults_cfg, file_cfg)
    return merged


def resolve_device(cfg: DictConfig) -> str:
    """Resolve device string, falling back to CPU if CUDA unavailable.

    Args:
        cfg: Config with a 'device' key.

    Returns:
        'cuda' if available, else 'cpu'.
    """
    import torch

    device = OmegaConf.to_container(cfg).get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def config_to_dict(cfg: DictConfig) -> dict[str, Any]:
    """Convert OmegaConf config to a plain Python dict.

    Args:
        cfg: OmegaConf DictConfig.

    Returns:
        Plain dict (safe for JSON serialization, wandb logging, etc.).
    """
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
