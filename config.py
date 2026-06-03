from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "batch_size": 8,
    "lr": 1e-4,
    "epochs": 100,
    "dataset_type": "dummy",
    "train_npz": "outputs/datasets/toy_fire_train.npz",
    "val_npz": "outputs/datasets/toy_fire_val.npz",
    "test_npz": "outputs/datasets/toy_fire_test.npz",
    "horizon": 5,
    "obs_window": 4,
    "channels": 6,
    "image_size": 64,
    "base_channels": 32,
    "latent_channels": 128,
    "gru_layers": 1,
    "dropout_p": 0.1,
    "num_mc_samples": 5,
    "uncertainty_alpha": 0.7,
    "num_trajectories": 64,
    "risk_threshold_delta": 1.2,
    "goal_weight": 0.3,
    "progress_weight": 0.2,
    "backtrack_penalty": 1.0,
    "trajectory_noise_scale": 4.0,
    "num_workers": 0,
    "amp": True,
    "grad_clip_norm": 1.0,
    "uncertainty_weight": 0.1,
    "lambda_u": 0.5,
    "log_interval": 10,
    "output_dir": "prism/outputs",
    "data": {
        "train_root": None,
        "val_root": None,
        "train_samples": 128,
        "val_samples": 32,
        "seed": 42,
    },
    "env": {
        "map_size": 64,
        "max_steps": 100,
        "num_fire_sources": 2,
        "obstacle_density": 0.05,
        "fire_spread_rate": 0.04,
        "smoke_spread_rate": 0.08,
        "risk_threshold_collision": 0.95,
        "goal_radius": 3.0,
        "max_step_size": 2.0,
    },
}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return _normalize_config(dict(DEFAULT_CONFIG), {})

    config_path = Path(path).expanduser()
    with config_path.open("r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}
    return _normalize_config(deep_update(DEFAULT_CONFIG, user_config), user_config)


def _normalize_config(config: dict[str, Any], user_config: dict[str, Any]) -> dict[str, Any]:
    if "in_channels" in config and "channels" not in user_config:
        config["channels"] = config["in_channels"]
    if "learning_rate" in config and "lr" not in user_config:
        config["lr"] = config["learning_rate"]
    if "alpha_unc" in config and "uncertainty_weight" not in user_config:
        config["uncertainty_weight"] = config["alpha_unc"]
    return config
