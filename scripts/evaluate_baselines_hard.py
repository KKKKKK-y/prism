from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = PROJECT_ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from prism.config import load_config
from prism.envs import HardToyFireEnv
from prism.planners import STAGE5_METHODS, build_planner_risk, sample_candidate_trajectories, select_safe_trajectory
from prism.scripts.evaluate_baselines import (
    EPISODE_FIELDS,
    MODEL_METHODS,
    SUMMARY_FIELDS,
    build_loaded_model,
    format_summary,
    needs_model,
    set_deterministic_seed,
    summarize_rows,
)
from prism.scripts.run_closed_loop import compute_path_length, resolve_device


ENV_KEYS = {
    "map_size",
    "max_steps",
    "obstacle_density",
    "fire_spread_rate",
    "smoke_spread_rate",
    "risk_threshold_collision",
}
PLANNER_KEYS = {
    "risk_threshold_delta",
    "goal_weight",
    "progress_weight",
    "backtrack_penalty",
    "num_trajectories",
    "trajectory_noise_scale",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage-5 methods on hard ToyFire scenarios.")
    parser.add_argument("--config", type=Path, default=Path("configs/toy_train.yaml"), help="Base model config.")
    parser.add_argument(
        "--hard-config",
        type=Path,
        default=Path("configs/toy_eval_hard.yaml"),
        help="Hard evaluation config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/checkpoints_toy/best.pt"),
        help="Path to trained PRISM checkpoint.",
    )
    parser.add_argument("--episodes", type=int, default=100, help="Number of hard episodes per method.")
    parser.add_argument("--methods", nargs="+", default=STAGE5_METHODS, help="Methods to evaluate.")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/results/stage5_hard_baseline_results.csv"),
        help="Hard benchmark summary CSV.",
    )
    parser.add_argument(
        "--episode-output",
        type=Path,
        default=Path("outputs/results/stage5_hard_baseline_episode_results.csv"),
        help="Hard benchmark per-episode CSV.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Device to use for model inference.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-step debug metrics.")
    return parser.parse_args()


def load_hard_config(path: Path) -> dict[str, Any]:
    with path.expanduser().open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    hard = raw.get("hard_eval", raw)
    if not isinstance(hard, dict):
        raise ValueError(f"Hard config must contain a mapping, got {type(hard).__name__}")
    return hard


def apply_hard_eval_config(config: dict[str, Any], hard: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(config)
    env_cfg = dict(merged.get("env", {}))
    for key in ENV_KEYS:
        if key in hard:
            env_cfg[key] = hard[key]
    merged["env"] = env_cfg
    for key in PLANNER_KEYS:
        if key in hard:
            merged[key] = hard[key]
    return merged


def build_hard_env(config: dict[str, Any], hard: dict[str, Any], seed: int | None) -> HardToyFireEnv:
    env_cfg = config.get("env", {})
    return HardToyFireEnv(
        map_size=int(env_cfg.get("map_size", config.get("image_size", 64))),
        obs_window=int(config.get("obs_window", 4)),
        channels=int(config.get("channels", 5)),
        max_steps=int(env_cfg.get("max_steps", 90)),
        num_fire_sources=int(hard.get("fire_source_count_max", env_cfg.get("num_fire_sources", 5))),
        obstacle_density=float(env_cfg.get("obstacle_density", 0.12)),
        fire_spread_rate=float(env_cfg.get("fire_spread_rate", 0.08)),
        smoke_spread_rate=float(env_cfg.get("smoke_spread_rate", 0.14)),
        risk_threshold_collision=float(env_cfg.get("risk_threshold_collision", 0.70)),
        goal_radius=float(env_cfg.get("goal_radius", 3.0)),
        max_step_size=float(env_cfg.get("max_step_size", 2.0)),
        seed=seed,
        hard_eval=hard,
    )


def run_hard_episode(
    *,
    method: str,
    config: dict[str, Any],
    hard: dict[str, Any],
    device: torch.device,
    seed: int,
    model: torch.nn.Module | None,
    verbose: bool = False,
) -> dict[str, Any]:
    env = build_hard_env(config, hard, seed=seed)
    observation = env.reset(seed=seed)

    horizon = int(config.get("horizon", 5))
    map_size = int(config.get("env", {}).get("map_size", config.get("image_size", 64)))
    num_trajectories = int(config.get("num_trajectories", 64))
    delta = float(config.get("risk_threshold_delta", 0.9))
    goal_weight = float(config.get("goal_weight", 0.3))
    noise_scale = float(config.get("trajectory_noise_scale", 5.0))
    progress_weight = float(config.get("progress_weight", 0.2))
    backtrack_penalty = float(config.get("backtrack_penalty", 1.0))
    max_step_size = float(config.get("env", {}).get("max_step_size", 2.0))

    path = [observation["robot_xy"].clone()]
    cumulative_risk = 0.0
    done = False
    info: dict[str, Any] = {"success": False, "collision": False, "timeout": False, "risk_at_robot": 0.0}

    while not done:
        step_seed = int(seed * 1009 + int(info.get("timestep", 0)) * 9176 + 17)
        set_deterministic_seed(step_seed)
        planner_risk, risk_info = build_planner_risk(method, observation, model, config, device)

        robot_xy = observation["robot_xy"]
        goal_xy = observation["goal_xy"]
        set_deterministic_seed(step_seed + 1)
        trajectories = sample_candidate_trajectories(
            start_xy=robot_xy,
            goal_xy=goal_xy,
            num_trajectories=num_trajectories,
            horizon=horizon,
            map_size=map_size,
            noise_scale=noise_scale,
            max_step_size=max_step_size,
        )
        best_traj, _, _, _, planner_info = select_safe_trajectory(
            safe_risk_map=planner_risk,
            trajectories=trajectories,
            goal_xy=goal_xy,
            start_xy=robot_xy,
            delta=delta,
            goal_weight=goal_weight,
            progress_weight=progress_weight,
            backtrack_penalty=backtrack_penalty,
        )

        observation, done, info = env.step(best_traj[0])
        path.append(observation["robot_xy"].clone())
        cumulative_risk += float(info["risk_at_robot"])

        if verbose:
            print(
                f"method={method} step={info['timestep']} "
                f"risk_method={risk_info['method']} "
                f"distance_to_goal={info['distance_to_goal']:.3f} "
                f"current_risk={info['risk_at_robot']:.4f} "
                f"best_risk={planner_info['best_risk']:.4f} "
                f"num_feasible={planner_info['num_feasible']} "
                f"collision={info['collision']} done={done}"
            )

    path_length = float(info.get("path_length", compute_path_length(path)))
    failure_reason = "reached_goal" if bool(info["success"]) else str(info.get("failure_reason", "none"))
    return {
        "success": int(bool(info["success"])),
        "collision": int(bool(info["collision"])),
        "timeout": int(bool(info["timeout"])),
        "failure_reason": failure_reason,
        "cumulative_risk": cumulative_risk,
        "path_length": path_length,
        "steps": int(info["timestep"]),
    }


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError(f"--episodes must be positive, got {args.episodes}")
    methods = list(dict.fromkeys(args.methods))
    invalid = [method for method in methods if method not in STAGE5_METHODS]
    if invalid:
        raise ValueError(f"Unknown methods {invalid}. Valid methods: {STAGE5_METHODS}")

    base_config = load_config(args.config)
    hard = load_hard_config(args.hard_config)
    config = apply_hard_eval_config(base_config, hard)
    base_seed = int(config.get("data", {}).get("seed", config.get("seed", 42)))
    seeds = [base_seed + idx for idx in range(args.episodes)]
    device = resolve_device(args.device)
    model = build_loaded_model(config, args.checkpoint, device) if needs_model(methods) else None

    all_episode_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    print("Stage-5 hard methods:", " ".join(methods))
    print("Hard config:", args.hard_config)
    print("Seed list:", seeds)
    for method in methods:
        method_rows: list[dict[str, Any]] = []
        for episode_idx, seed in enumerate(seeds):
            metrics = run_hard_episode(
                method=method,
                config=config,
                hard=hard,
                device=device,
                seed=seed,
                model=model if method in MODEL_METHODS else None,
                verbose=args.verbose,
            )
            row = {
                "method": method,
                "episode": episode_idx,
                "seed": seed,
                **metrics,
            }
            method_rows.append(row)
            all_episode_rows.append(row)
            print(
                f"method={method} episode={episode_idx + 1}/{args.episodes} "
                f"seed={seed} success={bool(metrics['success'])} "
                f"collision={bool(metrics['collision'])} timeout={bool(metrics['timeout'])} "
                f"steps={metrics['steps']} cumulative_risk={metrics['cumulative_risk']:.4f}"
            )

        summary = summarize_rows(method, method_rows)
        summary_rows.append(summary)
        print(format_summary(summary))

    summary_output = args.summary_output.expanduser()
    episode_output = args.episode_output.expanduser()
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    episode_output.parent.mkdir(parents=True, exist_ok=True)

    with summary_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)

    with episode_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EPISODE_FIELDS)
        writer.writeheader()
        writer.writerows(all_episode_rows)

    print("Stage-5 hard baseline summary:")
    for row in summary_rows:
        print(format_summary(row))
    print(f"Saved hard baseline summary to: {summary_output}")
    print(f"Saved hard baseline episode results to: {episode_output}")


if __name__ == "__main__":
    main()
