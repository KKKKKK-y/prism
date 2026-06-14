from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = PROJECT_ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from prism.scripts.diagnose_hard_planner import (
    build_diagnostic_risk,
    load_stage5_2_config,
    score_candidate_trajectories,
)
from prism.scripts.evaluate_baselines import MODEL_METHODS, build_loaded_model, set_deterministic_seed
from prism.scripts.evaluate_baselines_hard import build_hard_env
from prism.scripts.run_closed_loop import resolve_device
from prism.planners import sample_candidate_trajectories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check planner trajectory and ToyFireEnv step/risk alignment.")
    parser.add_argument("--config", type=Path, default=Path("configs/toy_train_hard_level_b.yaml"))
    parser.add_argument("--hard-config", type=Path, default=Path("configs/toy_eval_hard.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints_toy_hard_b/best.pt"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", default="prism_full")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Device to use for model inference.",
    )
    return parser.parse_args()


def _map_value(risk_map: torch.Tensor, xy: torch.Tensor, *, swapped: bool = False) -> float:
    if risk_map.ndim == 4:
        risk_map = risk_map[:, 0]
    if risk_map.ndim == 3:
        risk_map = risk_map[0]
    height, width = int(risk_map.shape[-2]), int(risk_map.shape[-1])
    x = int(round(float(xy[0].item())))
    y = int(round(float(xy[1].item())))
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    if swapped:
        return float(risk_map[x, y].item())
    return float(risk_map[y, x].item())


def _print_traj(name: str, trajectory: torch.Tensor) -> None:
    coords = []
    for point in trajectory.detach().cpu():
        coords.append(f"({float(point[0].item()):.3f},{float(point[1].item()):.3f})")
    print(f"{name}: " + " ".join(coords))


def main() -> None:
    args = parse_args()
    config, hard = load_stage5_2_config(args.config, args.hard_config)
    device = resolve_device(args.device)
    model = build_loaded_model(config, args.checkpoint, device) if args.method in MODEL_METHODS else None

    env = build_hard_env(config, hard, seed=args.seed)
    observation = env.reset(seed=args.seed)
    robot_before = observation["robot_xy"].clone()
    goal_xy = observation["goal_xy"].clone()
    step_seed = int(args.seed * 1009 + 17)

    set_deterministic_seed(step_seed)
    planner_risk, risk_stats = build_diagnostic_risk(args.method, observation, model, config, device)

    horizon = int(config.get("horizon", 5))
    map_size = int(config.get("env", {}).get("map_size", config.get("image_size", 64)))
    num_trajectories = int(config.get("num_trajectories", 64))
    noise_scale = float(config.get("trajectory_noise_scale", 5.0))
    max_step_size = float(config.get("env", {}).get("max_step_size", 2.0))

    set_deterministic_seed(step_seed + 1)
    trajectories = sample_candidate_trajectories(
        start_xy=robot_before,
        goal_xy=goal_xy,
        num_trajectories=num_trajectories,
        horizon=horizon,
        map_size=map_size,
        noise_scale=noise_scale,
        max_step_size=max_step_size,
    )
    candidate_info = score_candidate_trajectories(
        safe_risk_map=planner_risk,
        trajectories=trajectories,
        start_xy=robot_before,
        goal_xy=goal_xy,
        delta=float(config.get("risk_threshold_delta", 1.4)),
        goal_weight=float(config.get("goal_weight", 0.3)),
        progress_weight=float(config.get("progress_weight", 0.2)),
        backtrack_penalty=float(config.get("backtrack_penalty", 1.0)),
    )
    best_idx = int(candidate_info["best_idx"])
    selected_traj = trajectories[best_idx]
    action_sent_to_env = selected_traj[0].clone()
    planner_sampled_risk = float(candidate_info["trajectory_risks"][best_idx].item())
    planner_h0_yx = _map_value(planner_risk[0], action_sent_to_env, swapped=False)
    planner_h0_xy = _map_value(planner_risk[0], action_sent_to_env, swapped=True)

    current_before_yx = _map_value(observation["risk_map"], robot_before, swapped=False)
    current_before_xy = _map_value(observation["risk_map"], robot_before, swapped=True)
    action_target_distance = float(torch.linalg.norm(action_sent_to_env - robot_before).item())

    observation_after, done, info = env.step(action_sent_to_env)
    robot_after = observation_after["robot_xy"].clone()
    env_after_yx = _map_value(observation_after["risk_map"], robot_after, swapped=False)
    env_after_xy = _map_value(observation_after["risk_map"], robot_after, swapped=True)
    planner_h0_after_yx = _map_value(planner_risk[0], robot_after, swapped=False)
    planner_h0_after_xy = _map_value(planner_risk[0], robot_after, swapped=True)

    print("Planner / env alignment check")
    print("-----------------------------")
    print(f"method: {args.method}")
    print(f"seed: {args.seed}")
    print(f"horizon: {horizon}")
    print(f"max_step_size: {max_step_size:.6f}")
    print(f"num_candidates: {num_trajectories}")
    print(f"num_feasible: {int(candidate_info['num_feasible'])}")
    print(f"feasible_ratio: {int(candidate_info['num_feasible']) / max(1, num_trajectories):.6f}")
    print(f"risk_stats_safe_mean/p95/max: {risk_stats['safe_risk_mean']:.6f} / {risk_stats['safe_risk_p95']:.6f} / {risk_stats['safe_risk_max']:.6f}")
    print(f"robot_xy_before: ({float(robot_before[0]):.6f}, {float(robot_before[1]):.6f})")
    print(f"goal_xy: ({float(goal_xy[0]):.6f}, {float(goal_xy[1]):.6f})")
    _print_traj("selected_trajectory_xy", selected_traj)
    print(f"action_sent_to_env: ({float(action_sent_to_env[0]):.6f}, {float(action_sent_to_env[1]):.6f})")
    print(f"action_target_distance_from_robot: {action_target_distance:.6f}")
    print(f"planner_sampled_trajectory_risk_sum: {planner_sampled_risk:.6f}")
    print(f"planner_h0_at_action_yx: {planner_h0_yx:.6f}")
    print(f"planner_h0_at_action_xy_swapped: {planner_h0_xy:.6f}")
    print(f"current_env_risk_before_at_robot_yx: {current_before_yx:.6f}")
    print(f"current_env_risk_before_at_robot_xy_swapped: {current_before_xy:.6f}")
    print(f"robot_xy_after: ({float(robot_after[0]):.6f}, {float(robot_after[1]):.6f})")
    print(f"actual_action_distance: {float(info['action_distance']):.6f}")
    print(f"env_after_risk_at_robot_yx: {env_after_yx:.6f}")
    print(f"env_after_risk_at_robot_xy_swapped: {env_after_xy:.6f}")
    print(f"planner_h0_at_robot_after_yx: {planner_h0_after_yx:.6f}")
    print(f"planner_h0_at_robot_after_xy_swapped: {planner_h0_after_xy:.6f}")
    print(f"env_info_risk_at_robot: {float(info['risk_at_robot']):.6f}")
    print(f"collision: {bool(info['collision'])}")
    print(f"success: {bool(info['success'])}")
    print(f"done: {bool(done)}")
    print(f"failure_reason: {info.get('failure_reason', 'none')}")


if __name__ == "__main__":
    main()
