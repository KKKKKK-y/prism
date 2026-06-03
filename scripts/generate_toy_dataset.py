from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from prism.config import load_config
from prism.envs import ToyFireEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate supervised ToyFireEnv risk prediction dataset.")
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"), help="Path to YAML config.")
    parser.add_argument("--train_episodes", type=int, default=100, help="Number of train episodes.")
    parser.add_argument("--val_episodes", type=int, default=20, help="Number of validation episodes.")
    parser.add_argument("--test_episodes", type=int, default=20, help="Number of test episodes.")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/datasets"), help="Directory for output npz files.")
    return parser.parse_args()


def build_env(config: dict, seed: int) -> ToyFireEnv:
    env_cfg = config.get("env", {})
    return ToyFireEnv(
        map_size=int(env_cfg.get("map_size", 64)),
        obs_window=int(config.get("obs_window", 4)),
        channels=5,
        max_steps=int(env_cfg.get("max_steps", 100)),
        num_fire_sources=int(env_cfg.get("num_fire_sources", 2)),
        obstacle_density=float(env_cfg.get("obstacle_density", 0.05)),
        fire_spread_rate=float(env_cfg.get("fire_spread_rate", 0.04)),
        smoke_spread_rate=float(env_cfg.get("smoke_spread_rate", 0.08)),
        risk_threshold_collision=float(env_cfg.get("risk_threshold_collision", 0.95)),
        goal_radius=float(env_cfg.get("goal_radius", 3.0)),
        max_step_size=float(env_cfg.get("max_step_size", 2.0)),
        seed=seed,
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


def generate_split(config: dict, episodes: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    obs_window = int(config.get("obs_window", 4))
    horizon = int(config.get("horizon", 5))
    max_steps = int(config.get("env", {}).get("max_steps", 100))
    generator = torch.Generator().manual_seed(seed)
    obs_samples: list[np.ndarray] = []
    target_samples: list[np.ndarray] = []

    for episode_idx in range(episodes):
        env = build_env(config, seed=seed + episode_idx)
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


def save_split(path: Path, obs: np.ndarray, target: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, obs=obs.astype(np.float32, copy=False), target=target.astype(np.float32, copy=False))


def print_shapes(name: str, obs: np.ndarray, target: np.ndarray) -> None:
    print(f"{name} obs shape: {obs.shape}")
    print(f"{name} target shape: {target.shape}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = args.output_dir.expanduser()
    base_seed = int(config.get("seed", config.get("data", {}).get("seed", 42)))

    train_obs, train_target = generate_split(config, args.train_episodes, base_seed)
    val_obs, val_target = generate_split(config, args.val_episodes, base_seed + 10_000)
    test_obs, test_target = generate_split(config, args.test_episodes, base_seed + 20_000)

    save_split(output_dir / "toy_fire_train.npz", train_obs, train_target)
    save_split(output_dir / "toy_fire_val.npz", val_obs, val_target)
    save_split(output_dir / "toy_fire_test.npz", test_obs, test_target)

    print_shapes("train", train_obs, train_target)
    print_shapes("val", val_obs, val_target)
    print_shapes("test", test_obs, test_target)


if __name__ == "__main__":
    main()
