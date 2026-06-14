from __future__ import annotations

import argparse
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
from prism.planners import build_fused_risk_map, build_planner_risk, sample_candidate_trajectories, select_safe_trajectory
from prism.scripts.evaluate_baselines import build_loaded_model, set_deterministic_seed
from prism.scripts.evaluate_baselines_hard import apply_hard_eval_config, build_hard_env, load_hard_config
from prism.scripts.run_closed_loop import compute_path_length, resolve_device


FINAL_METHODS = [
    "goal_greedy",
    "current_only",
    "predicted_only",
    "max_fusion",
    "alpha_fusion_0.4",
]

SUMMARY_FIELDS = [
    "method",
    "episodes",
    "success_rate",
    "collision_rate",
    "timeout_rate",
    "avg_cumulative_risk",
    "avg_path_length",
    "avg_steps",
]

EPISODE_FIELDS = [
    "method",
    "episode",
    "seed",
    "success",
    "collision",
    "timeout",
    "failure_reason",
    "cumulative_risk",
    "path_length",
    "steps",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-5.9 final PRISM fusion planner evaluation.")
    parser.add_argument("--config", type=Path, default=Path("configs/prism_final_fusion.yaml"))
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/results/stage5_9_final_fusion_results.csv"),
    )
    parser.add_argument(
        "--episode-output",
        type=Path,
        default=Path("outputs/results/stage5_9_final_fusion_episode_results.csv"),
    )
    return parser.parse_args()


def load_final_config(path: Path) -> dict[str, Any]:
    with path.expanduser().open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    for section in ("planner", "model", "eval"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"{path} must contain a '{section}' mapping")
    return config


def summarize(method: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    episodes = len(rows)
    if episodes <= 0:
        raise ValueError(f"No rows to summarize for method={method}")
    return {
        "method": method,
        "episodes": episodes,
        "success_rate": sum(int(row["success"]) for row in rows) / episodes,
        "collision_rate": sum(int(row["collision"]) for row in rows) / episodes,
        "timeout_rate": sum(int(row["timeout"]) for row in rows) / episodes,
        "avg_cumulative_risk": sum(float(row["cumulative_risk"]) for row in rows) / episodes,
        "avg_path_length": sum(float(row["path_length"]) for row in rows) / episodes,
        "avg_steps": sum(float(row["steps"]) for row in rows) / episodes,
    }


def build_goal_greedy_risk(observation: dict[str, torch.Tensor], horizon: int) -> torch.Tensor:
    risk_map = observation["risk_map"].detach().float()
    return torch.zeros(horizon, 1, risk_map.shape[0], risk_map.shape[1], dtype=torch.float32)


def build_final_planner_risk(
    *,
    method: str,
    observation: dict[str, torch.Tensor],
    model: torch.nn.Module,
    config: dict[str, Any],
    device: torch.device,
    alpha: float,
) -> torch.Tensor:
    horizon = int(config.get("horizon", 5))
    if method == "goal_greedy":
        return build_goal_greedy_risk(observation, horizon)
    if method == "current_only":
        return build_fused_risk_map(
            current_risk=observation["risk_map"],
            predicted_risk=build_goal_greedy_risk(observation, horizon),
            mode="current_only",
        )

    predicted_safe_risk, _ = build_planner_risk("prism_full", observation, model, config, device)
    if method == "predicted_only":
        return build_fused_risk_map(observation["risk_map"], predicted_safe_risk, mode="predicted_only")
    if method == "max_fusion":
        return build_fused_risk_map(observation["risk_map"], predicted_safe_risk, mode="max_fusion")
    if method == "alpha_fusion_0.4":
        return build_fused_risk_map(
            observation["risk_map"],
            predicted_safe_risk,
            mode="alpha_fusion",
            alpha=alpha,
        )
    raise ValueError(f"Unknown final method: {method}")


def run_episode(
    *,
    method: str,
    config: dict[str, Any],
    hard: dict[str, Any],
    device: torch.device,
    seed: int,
    model: torch.nn.Module,
    alpha: float,
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
        planner_risk = build_final_planner_risk(
            method=method,
            observation=observation,
            model=model,
            config=config,
            device=device,
            alpha=alpha,
        )

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
        best_traj, _, _, _, _ = select_safe_trajectory(
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
    final_config = load_final_config(args.config)
    planner_cfg = final_config["planner"]
    model_cfg = final_config["model"]
    eval_cfg = final_config["eval"]

    episodes = int(args.episodes if args.episodes is not None else eval_cfg.get("episodes", 100))
    if episodes <= 0:
        raise ValueError(f"--episodes must be positive, got {episodes}")

    alpha = float(planner_cfg.get("alpha", 0.4))
    model_config_path = Path(model_cfg["config"])
    checkpoint_path = Path(model_cfg["checkpoint"])
    hard_config_path = Path(eval_cfg.get("hard_config", "configs/toy_eval_hard.yaml"))

    base_config = load_config(model_config_path)
    base_config["lambda_u"] = float(planner_cfg.get("lambda_u", base_config.get("lambda_u", 0.5)))
    base_config["uncertainty_alpha"] = float(planner_cfg.get("uncertainty_alpha", base_config.get("uncertainty_alpha", 0.7)))
    base_config["num_mc_samples"] = int(planner_cfg.get("num_mc_samples", base_config.get("num_mc_samples", 5)))
    hard = load_hard_config(hard_config_path)
    config = apply_hard_eval_config(base_config, hard)
    base_seed = int(config.get("data", {}).get("seed", config.get("seed", 42)))
    seeds = [base_seed + idx for idx in range(episodes)]
    device = resolve_device(args.device)
    model = build_loaded_model(config, checkpoint_path, device)

    summary_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    print(
        "Stage-5.9 final fusion eval: "
        f"config={model_config_path} checkpoint={checkpoint_path} hard_config={hard_config_path} "
        f"episodes={episodes} alpha={alpha}"
    )
    for method in FINAL_METHODS:
        method_rows = []
        for episode_idx, seed in enumerate(seeds):
            metrics = run_episode(
                method=method,
                config=config,
                hard=hard,
                device=device,
                seed=seed,
                model=model,
                alpha=alpha,
            )
            row = {"method": method, "episode": episode_idx, "seed": seed, **metrics}
            method_rows.append(row)
            episode_rows.append(row)
            print(
                f"method={method} episode={episode_idx + 1}/{episodes} seed={seed} "
                f"success={bool(metrics['success'])} collision={bool(metrics['collision'])} "
                f"timeout={bool(metrics['timeout'])} steps={metrics['steps']} "
                f"cumulative_risk={metrics['cumulative_risk']:.4f}",
                flush=True,
            )
        summary = summarize(method, method_rows)
        summary_rows.append(summary)
        print(
            f"{method}: success_rate={summary['success_rate']:.6f} "
            f"collision_rate={summary['collision_rate']:.6f} timeout_rate={summary['timeout_rate']:.6f} "
            f"avg_cumulative_risk={summary['avg_cumulative_risk']:.6f} "
            f"avg_path_length={summary['avg_path_length']:.6f} avg_steps={summary['avg_steps']:.6f}",
            flush=True,
        )

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.episode_output.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)
    with args.episode_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EPISODE_FIELDS)
        writer.writeheader()
        writer.writerows(episode_rows)
    print(f"Saved Stage-5.9 final summary to: {args.summary_output}")
    print(f"Saved Stage-5.9 final episode results to: {args.episode_output}")


if __name__ == "__main__":
    main()
