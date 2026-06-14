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
from prism.planners import FUSION_MODES, build_fused_risk_map, build_planner_risk, sample_candidate_trajectories, select_safe_trajectory
from prism.scripts.evaluate_baselines import EPISODE_FIELDS, SUMMARY_FIELDS, build_loaded_model, format_summary, set_deterministic_seed, summarize_rows
from prism.scripts.evaluate_baselines_hard import apply_hard_eval_config, build_hard_env, load_hard_config
from prism.scripts.run_closed_loop import compute_path_length, resolve_device


FUSION_SUMMARY_FIELDS = [
    "mode",
    "alpha",
    "scale",
    "episodes",
    "success_rate",
    "collision_rate",
    "timeout_rate",
    "avg_cumulative_risk",
    "avg_path_length",
    "avg_steps",
]
FUSION_EPISODE_FIELDS = [
    "mode",
    "alpha",
    "scale",
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


def parse_optional_float(value: str) -> float | None:
    if value.upper() == "NA":
        return None
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage-5.8 current/predicted fusion planner.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--hard-config", type=Path, default=Path("configs/toy_eval_hard.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--modes", nargs="+", default=FUSION_MODES)
    parser.add_argument("--alpha", type=parse_optional_float, default=None)
    parser.add_argument("--scale", type=parse_optional_float, default=None)
    parser.add_argument("--summary-output", type=Path, default=Path("outputs/results/stage5_8_fusion_results.csv"))
    parser.add_argument("--episode-output", type=Path, default=Path("outputs/results/stage5_8_fusion_episode_results.csv"))
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def display_value(value: float | None) -> str:
    return "NA" if value is None else f"{value:g}"


def run_fusion_episode(
    *,
    mode: str,
    alpha: float | None,
    scale: float | None,
    config: dict[str, Any],
    hard: dict[str, Any],
    device: torch.device,
    seed: int,
    model: torch.nn.Module,
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
        predicted_risk, risk_info = build_planner_risk("prism_full", observation, model, config, device)
        planner_risk = build_fused_risk_map(
            current_risk=observation["risk_map"],
            predicted_risk=predicted_risk,
            mode=mode,
            alpha=0.5 if alpha is None else alpha,
            scale=1.0 if scale is None else scale,
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
                f"mode={mode} alpha={display_value(alpha)} scale={display_value(scale)} "
                f"step={info['timestep']} risk_method={risk_info['method']} "
                f"distance_to_goal={info['distance_to_goal']:.3f} current_risk={info['risk_at_robot']:.4f} "
                f"best_risk={planner_info['best_risk']:.4f} collision={info['collision']} done={done}"
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


def summarize_fusion_rows(mode: str, alpha: float | None, scale: float | None, rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_rows(mode, rows)
    return {
        "mode": mode,
        "alpha": display_value(alpha),
        "scale": display_value(scale),
        "episodes": summary["episodes"],
        "success_rate": summary["success_rate"],
        "collision_rate": summary["collision_rate"],
        "timeout_rate": summary["timeout_rate"],
        "avg_cumulative_risk": summary["avg_cumulative_risk"],
        "avg_path_length": summary["avg_path_length"],
        "avg_steps": summary["avg_steps"],
    }


def evaluate_fusion_setting(
    *,
    mode: str,
    alpha: float | None,
    scale: float | None,
    config: dict[str, Any],
    hard: dict[str, Any],
    seeds: list[int],
    device: torch.device,
    model: torch.nn.Module,
    verbose: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    for episode_idx, seed in enumerate(seeds):
        metrics = run_fusion_episode(
            mode=mode,
            alpha=alpha,
            scale=scale,
            config=config,
            hard=hard,
            device=device,
            seed=seed,
            model=model,
            verbose=verbose,
        )
        row = {
            "mode": mode,
            "alpha": display_value(alpha),
            "scale": display_value(scale),
            "episode": episode_idx,
            "seed": seed,
            **metrics,
        }
        rows.append(row)
        print(
            f"mode={mode} alpha={display_value(alpha)} scale={display_value(scale)} "
            f"episode={episode_idx + 1}/{len(seeds)} seed={seed} "
            f"success={bool(metrics['success'])} collision={bool(metrics['collision'])} "
            f"timeout={bool(metrics['timeout'])} steps={metrics['steps']} "
            f"cumulative_risk={metrics['cumulative_risk']:.4f}"
        )
    summary = summarize_fusion_rows(mode, alpha, scale, rows)
    print(
        f"{mode} alpha={display_value(alpha)} scale={display_value(scale)}: "
        f"success_rate={float(summary['success_rate']):.6f} "
        f"collision_rate={float(summary['collision_rate']):.6f} "
        f"timeout_rate={float(summary['timeout_rate']):.6f} "
        f"avg_cumulative_risk={float(summary['avg_cumulative_risk']):.6f} "
        f"avg_path_length={float(summary['avg_path_length']):.6f} "
        f"avg_steps={float(summary['avg_steps']):.6f}"
    )
    return summary, rows


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError(f"--episodes must be positive, got {args.episodes}")
    modes = list(dict.fromkeys(args.modes))
    invalid = [mode for mode in modes if mode not in FUSION_MODES]
    if invalid:
        raise ValueError(f"Unknown fusion modes {invalid}. Valid modes: {FUSION_MODES}")

    base_config = load_config(args.config)
    hard = load_hard_config(args.hard_config)
    config = apply_hard_eval_config(base_config, hard)
    base_seed = int(config.get("data", {}).get("seed", config.get("seed", 42)))
    seeds = [base_seed + idx for idx in range(args.episodes)]
    device = resolve_device(args.device)
    model = build_loaded_model(config, args.checkpoint, device)

    summary_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for mode in modes:
        alpha = args.alpha if mode == "alpha_fusion" else None
        scale = args.scale if mode in {"calibrated_predicted", "max_calibrated_fusion"} else None
        summary, rows = evaluate_fusion_setting(
            mode=mode,
            alpha=alpha,
            scale=scale,
            config=config,
            hard=hard,
            seeds=seeds,
            device=device,
            model=model,
            verbose=args.verbose,
        )
        summary_rows.append(summary)
        episode_rows.extend(rows)

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.episode_output.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FUSION_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)
    with args.episode_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FUSION_EPISODE_FIELDS)
        writer.writeheader()
        writer.writerows(episode_rows)
    print(f"Saved fusion summary to: {args.summary_output}")
    print(f"Saved fusion episode results to: {args.episode_output}")


if __name__ == "__main__":
    main()
