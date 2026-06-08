from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = PROJECT_ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from prism.config import load_config
from prism.planners import STAGE5_METHODS, build_planner_risk, sample_candidate_trajectories, select_safe_trajectory
from prism.scripts.run_closed_loop import build_env, build_model, compute_path_length, load_checkpoint_if_available, resolve_device


MODEL_METHODS = {"mean_risk", "prism_no_propagation", "prism_full"}
SUMMARY_FIELDS = [
    "method",
    "episodes",
    "success_rate",
    "collision_rate",
    "timeout_rate",
    "avg_cumulative_risk",
    "avg_path_length",
    "avg_steps",
    "reached_goal",
    "collision_high_risk",
    "collision_obstacle",
    "timeout",
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
    parser = argparse.ArgumentParser(description="Evaluate PRISM Stage-5 baseline and ablation methods.")
    parser.add_argument("--config", type=Path, default=Path("configs/toy_train.yaml"), help="Path to YAML config.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/checkpoints_toy/best.pt"),
        help="Path to trained PRISM checkpoint.",
    )
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes per method.")
    parser.add_argument("--methods", nargs="+", default=STAGE5_METHODS, help="Methods to evaluate.")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/results/stage5_baseline_results.csv"),
        help="Summary CSV output path.",
    )
    parser.add_argument(
        "--episode-output",
        type=Path,
        default=Path("outputs/results/stage5_baseline_episode_results.csv"),
        help="Per-episode CSV output path.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Device to use for model inference.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-step debug metrics.")
    return parser.parse_args()


def set_deterministic_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def needs_model(methods: list[str]) -> bool:
    return any(method in MODEL_METHODS for method in methods)


def build_loaded_model(config: dict[str, Any], checkpoint: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = checkpoint.expanduser()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Stage-5 model methods require checkpoint, but it was not found: {checkpoint}")
    model = build_model(config).to(device)
    load_checkpoint_if_available(model, checkpoint, device)
    model.eval()
    return model


def summarize_rows(method: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    episodes = len(rows)
    if episodes == 0:
        raise ValueError(f"No episode rows for method {method}")
    counts = {
        "reached_goal": 0,
        "collision_high_risk": 0,
        "collision_obstacle": 0,
        "timeout": 0,
    }
    for row in rows:
        reason = str(row["failure_reason"])
        if reason in counts:
            counts[reason] += 1

    return {
        "method": method,
        "episodes": episodes,
        "success_rate": sum(int(row["success"]) for row in rows) / episodes,
        "collision_rate": sum(int(row["collision"]) for row in rows) / episodes,
        "timeout_rate": sum(int(row["timeout"]) for row in rows) / episodes,
        "avg_cumulative_risk": sum(float(row["cumulative_risk"]) for row in rows) / episodes,
        "avg_path_length": sum(float(row["path_length"]) for row in rows) / episodes,
        "avg_steps": sum(float(row["steps"]) for row in rows) / episodes,
        **counts,
    }


def run_baseline_episode(
    *,
    method: str,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
    model: torch.nn.Module | None,
    verbose: bool = False,
) -> dict[str, Any]:
    env = build_env(config, seed=seed)
    observation = env.reset(seed=seed)

    horizon = int(config.get("horizon", 5))
    map_size = int(config.get("env", {}).get("map_size", config.get("image_size", 64)))
    num_trajectories = int(config.get("num_trajectories", 64))
    delta = float(config.get("risk_threshold_delta", 0.8))
    goal_weight = float(config.get("goal_weight", 0.1))
    noise_scale = float(config.get("trajectory_noise_scale", 4.0))
    progress_weight = float(config.get("progress_weight", 0.2))
    backtrack_penalty = float(config.get("backtrack_penalty", 1.0))

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


def format_summary(row: dict[str, Any]) -> str:
    return (
        f"{row['method']}: "
        f"success_rate={float(row['success_rate']):.6f} "
        f"collision_rate={float(row['collision_rate']):.6f} "
        f"timeout_rate={float(row['timeout_rate']):.6f} "
        f"avg_cumulative_risk={float(row['avg_cumulative_risk']):.6f} "
        f"avg_path_length={float(row['avg_path_length']):.6f} "
        f"avg_steps={float(row['avg_steps']):.6f}"
    )


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError(f"--episodes must be positive, got {args.episodes}")
    methods = list(dict.fromkeys(args.methods))
    invalid = [method for method in methods if method not in STAGE5_METHODS]
    if invalid:
        raise ValueError(f"Unknown methods {invalid}. Valid methods: {STAGE5_METHODS}")

    config = load_config(args.config)
    base_seed = int(config.get("data", {}).get("seed", config.get("seed", 42)))
    seeds = [base_seed + idx for idx in range(args.episodes)]
    device = resolve_device(args.device)
    model = build_loaded_model(config, args.checkpoint, device) if needs_model(methods) else None

    all_episode_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    print("Stage-5 methods:", " ".join(methods))
    print("Seed list:", seeds)
    for method in methods:
        method_rows: list[dict[str, Any]] = []
        for episode_idx, seed in enumerate(seeds):
            metrics = run_baseline_episode(
                method=method,
                config=config,
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

    print("Stage-5 baseline summary:")
    for row in summary_rows:
        print(format_summary(row))
    print(f"Saved baseline summary to: {summary_output}")
    print(f"Saved baseline episode results to: {episode_output}")


if __name__ == "__main__":
    main()
