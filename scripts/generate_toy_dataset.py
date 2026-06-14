from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from prism.config import load_config
from prism.envs import HardToyFireEnv, ToyFireEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate supervised ToyFireEnv risk prediction dataset.")
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"), help="Path to YAML config.")
    parser.add_argument("--train_episodes", type=int, default=100, help="Number of train episodes.")
    parser.add_argument("--val_episodes", type=int, default=20, help="Number of validation episodes.")
    parser.add_argument("--test_episodes", type=int, default=20, help="Number of test episodes.")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/datasets"), help="Directory for output npz files.")
    parser.add_argument("--mode", choices=("normal", "hard"), default="normal", help="Dataset generation mode.")
    parser.add_argument(
        "--hard-config",
        type=Path,
        default=Path("configs/toy_eval_hard.yaml"),
        help="Hard scenario config used when --mode hard.",
    )
    return parser.parse_args()


def load_hard_config(path: Path) -> dict:
    with path.expanduser().open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    hard = raw.get("hard_eval", raw)
    if not isinstance(hard, dict):
        raise ValueError(f"Hard config must contain a mapping, got {type(hard).__name__}")
    return hard


def apply_hard_config(config: dict, hard: dict) -> dict:
    updated = dict(config)
    env_cfg = dict(updated.get("env", {}))
    for key in (
        "map_size",
        "max_steps",
        "obstacle_density",
        "fire_spread_rate",
        "smoke_spread_rate",
        "risk_threshold_collision",
    ):
        if key in hard:
            env_cfg[key] = hard[key]
    updated["env"] = env_cfg
    for key in (
        "risk_threshold_delta",
        "goal_weight",
        "progress_weight",
        "backtrack_penalty",
        "num_trajectories",
        "trajectory_noise_scale",
    ):
        if key in hard:
            updated[key] = hard[key]
    return updated


def build_env(config: dict, seed: int, mode: str = "normal", hard_config: dict | None = None) -> ToyFireEnv:
    env_cfg = config.get("env", {})
    env_cls = HardToyFireEnv if mode == "hard" else ToyFireEnv
    extra_kwargs = {"hard_eval": hard_config or {}} if mode == "hard" else {}
    return env_cls(
        map_size=int(env_cfg.get("map_size", 64)),
        obs_window=int(config.get("obs_window", 4)),
        channels=5,
        max_steps=int(env_cfg.get("max_steps", 100)),
        num_fire_sources=int((hard_config or {}).get("fire_source_count_max", env_cfg.get("num_fire_sources", 2))),
        obstacle_density=float(env_cfg.get("obstacle_density", 0.05)),
        fire_spread_rate=float(env_cfg.get("fire_spread_rate", 0.04)),
        smoke_spread_rate=float(env_cfg.get("smoke_spread_rate", 0.08)),
        risk_threshold_collision=float(env_cfg.get("risk_threshold_collision", 0.95)),
        goal_radius=float(env_cfg.get("goal_radius", 3.0)),
        max_step_size=float(env_cfg.get("max_step_size", 2.0)),
        seed=seed,
        **extra_kwargs,
    )


def exploratory_action(env: ToyFireEnv, generator: torch.Generator) -> torch.Tensor:
    max_step = float(env.max_step_size)
    if torch.rand((), generator=generator).item() < 0.70:
        direction = env.goal_xy - env.robot_xy
    else:
        direction = torch.randn(2, generator=generator)

    distance = torch.linalg.norm(direction).item()
    if distance < 1e-8:
        direction = torch.randn(2, generator=generator)
        distance = torch.linalg.norm(direction).clamp_min(1e-8).item()
    return env.robot_xy + direction / distance * max_step


def generate_split(
    config: dict,
    episodes: int,
    seed: int,
    mode: str = "normal",
    hard_config: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    obs_window = int(config.get("obs_window", 4))
    horizon = int(config.get("horizon", 5))
    max_steps = int(config.get("env", {}).get("max_steps", 100))
    generator = torch.Generator().manual_seed(seed)
    obs_samples: list[np.ndarray] = []
    target_samples: list[np.ndarray] = []

    for episode_idx in range(episodes):
        env = build_env(config, seed=seed + episode_idx, mode=mode, hard_config=hard_config)
        observation = env.reset(seed=seed + episode_idx)
        frames = [observation["obs"][-1].clone()]
        risk_maps = [observation["risk_map"].clone()]

        for _ in range(max_steps):
            action_xy = exploratory_action(env, generator)
            observation, _, _ = env.step(action_xy)
            # frame: [C, 64, 64], channel 3 is current risk_map.
            frames.append(observation["obs"][-1].clone())
            # risk_map: [64, 64], map indexing uses [y, x].
            risk_maps.append(observation["risk_map"].clone())

        # obs: [k, C, 64, 64], target: [H, 1, 64, 64]
        for t in range(obs_window - 1, len(frames) - horizon):
            obs = torch.stack(frames[t - obs_window + 1 : t + 1], dim=0)
            target = torch.stack(risk_maps[t + 1 : t + horizon + 1], dim=0).unsqueeze(1)
            obs_samples.append(obs.numpy().astype(np.float32, copy=False))
            target_samples.append(target.numpy().astype(np.float32, copy=False))

    if not obs_samples:
        raise RuntimeError("No samples generated. Increase env.max_steps or episode count.")

    return np.stack(obs_samples).astype(np.float32), np.stack(target_samples).astype(np.float32)


def save_split(path: Path, obs: np.ndarray, target: np.ndarray, metadata: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "obs": obs.astype(np.float32, copy=False),
        "target": target.astype(np.float32, copy=False),
    }
    if metadata is not None:
        payload["metadata"] = np.array(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(path, **payload)


def print_shapes(name: str, obs: np.ndarray, target: np.ndarray) -> None:
    print(f"{name} obs shape: {obs.shape}")
    print(f"{name} target shape: {target.shape}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    hard_config = load_hard_config(args.hard_config) if args.mode == "hard" else None
    if hard_config is not None:
        config = apply_hard_config(config, hard_config)
    output_dir = args.output_dir.expanduser()
    base_seed = int(config.get("seed", config.get("data", {}).get("seed", 42)))

    train_obs, train_target = generate_split(config, args.train_episodes, base_seed, args.mode, hard_config)
    val_obs, val_target = generate_split(config, args.val_episodes, base_seed + 10_000, args.mode, hard_config)
    test_obs, test_target = generate_split(config, args.test_episodes, base_seed + 20_000, args.mode, hard_config)

    if args.mode == "hard":
        train_path = Path(config.get("train_npz", "outputs/datasets_hard/toy_fire_train_hard.npz"))
        val_path = Path(config.get("val_npz", "outputs/datasets_hard/toy_fire_val_hard.npz"))
        test_path = Path(config.get("test_npz", "outputs/datasets_hard/toy_fire_test_hard.npz"))
    else:
        train_path = output_dir / "toy_fire_train.npz"
        val_path = output_dir / "toy_fire_val.npz"
        test_path = output_dir / "toy_fire_test.npz"

    metadata = None
    if args.mode == "hard":
        metadata = {
            "mode": args.mode,
            "train_episodes": args.train_episodes,
            "val_episodes": args.val_episodes,
            "test_episodes": args.test_episodes,
            "hard_config": hard_config,
        }
    save_split(train_path, train_obs, train_target, metadata)
    save_split(val_path, val_obs, val_target, metadata)
    save_split(test_path, test_obs, test_target, metadata)

    print_shapes("train", train_obs, train_target)
    print_shapes("val", val_obs, val_target)
    print_shapes("test", test_obs, test_target)


if __name__ == "__main__":
    main()
