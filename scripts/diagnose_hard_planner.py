from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = PROJECT_ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from prism.config import load_config
from prism.planners import STAGE5_METHODS, evaluate_trajectory_risk, sample_candidate_trajectories
from prism.scripts.evaluate_baselines import MODEL_METHODS, build_loaded_model, needs_model, set_deterministic_seed
from prism.scripts.evaluate_baselines_hard import apply_hard_eval_config, build_hard_env, load_hard_config
from prism.scripts.run_closed_loop import compute_path_length, resolve_device
from prism.utils.uncertainty import compute_safe_risk, mc_dropout_predict, propagate_uncertainty


DIAGNOSTIC_FIELDS = [
    "episode",
    "seed",
    "method",
    "step",
    "robot_x",
    "robot_y",
    "goal_x",
    "goal_y",
    "distance_to_goal",
    "selected_action_x",
    "selected_action_y",
    "selected_traj_risk",
    "selected_final_distance",
    "selected_score",
    "num_candidates",
    "num_feasible",
    "feasible_ratio",
    "min_candidate_risk",
    "mean_candidate_risk",
    "max_candidate_risk",
    "min_goal_distance",
    "mean_goal_distance",
    "safe_risk_mean",
    "safe_risk_max",
    "safe_risk_p95",
    "mu_mean",
    "mu_max",
    "sigma_mean",
    "sigma_max",
    "collision",
    "success",
    "failure_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record detailed Stage-5.2 hard planner diagnostics.")
    parser.add_argument("--config", type=Path, default=Path("configs/toy_train_hard_level_b.yaml"))
    parser.add_argument("--hard-config", type=Path, default=Path("configs/toy_eval_hard.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints_toy_hard_b/best.pt"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--methods", nargs="+", default=STAGE5_METHODS)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/results/stage5_2_hard_planner_diagnostics.csv"),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Device to use for model inference.",
    )
    return parser.parse_args()


def _nan() -> float:
    return float("nan")


def _tensor_stats(tensor: torch.Tensor | None) -> dict[str, float]:
    if tensor is None:
        return {"mean": _nan(), "max": _nan(), "p95": _nan()}
    values = tensor.detach().float().cpu().flatten()
    if values.numel() == 0:
        return {"mean": _nan(), "max": _nan(), "p95": _nan()}
    return {
        "mean": float(values.mean().item()),
        "max": float(values.max().item()),
        "p95": float(torch.quantile(values, 0.95).item()),
    }


def build_diagnostic_risk(
    method: str,
    observation: dict[str, torch.Tensor],
    model: torch.nn.Module | None,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    horizon = int(config.get("horizon", 5))
    current_risk = observation["risk_map"].detach().float()
    height, width = int(current_risk.shape[0]), int(current_risk.shape[1])

    if method == "goal_greedy":
        planner_risk = torch.zeros(horizon, 1, height, width, dtype=torch.float32)
        return planner_risk, {
            "safe_risk_mean": 0.0,
            "safe_risk_max": 0.0,
            "safe_risk_p95": 0.0,
            "mu_mean": _nan(),
            "mu_max": _nan(),
            "sigma_mean": _nan(),
            "sigma_max": _nan(),
        }

    if method == "current_risk":
        planner_risk = current_risk.view(1, 1, height, width).repeat(horizon, 1, 1, 1)
        safe_stats = _tensor_stats(planner_risk)
        return planner_risk, {
            "safe_risk_mean": safe_stats["mean"],
            "safe_risk_max": safe_stats["max"],
            "safe_risk_p95": safe_stats["p95"],
            "mu_mean": _nan(),
            "mu_max": _nan(),
            "sigma_mean": _nan(),
            "sigma_max": _nan(),
        }

    if model is None:
        raise ValueError(f"Method {method!r} requires a loaded model.")

    obs = observation["obs"].unsqueeze(0).to(device)
    if method == "mean_risk":
        model.eval()
        with torch.no_grad():
            mu, _ = model(obs)
        planner_risk = mu[0].detach().cpu().float()
        safe_stats = _tensor_stats(planner_risk)
        mu_stats = _tensor_stats(planner_risk)
        return planner_risk, {
            "safe_risk_mean": safe_stats["mean"],
            "safe_risk_max": safe_stats["max"],
            "safe_risk_p95": safe_stats["p95"],
            "mu_mean": mu_stats["mean"],
            "mu_max": mu_stats["max"],
            "sigma_mean": _nan(),
            "sigma_max": _nan(),
        }

    num_samples = int(config.get("num_mc_samples", 5))
    lambda_u = float(config.get("lambda_u", 0.5))
    with torch.no_grad():
        mu_mean, sigma_mc, mu_samples = mc_dropout_predict(model, obs, num_samples=num_samples)

    if method == "prism_no_propagation":
        sigma = sigma_mc
    elif method == "prism_full":
        sigma, _ = propagate_uncertainty(mu_samples, alpha=float(config.get("uncertainty_alpha", 0.7)))
    else:
        raise ValueError(f"Unknown Stage-5 method {method!r}. Valid methods: {STAGE5_METHODS}")

    safe_risk = compute_safe_risk(mu_mean, sigma, lambda_u=lambda_u)[0].detach().cpu().float()
    mu_stats = _tensor_stats(mu_mean[0])
    sigma_stats = _tensor_stats(sigma[0])
    safe_stats = _tensor_stats(safe_risk)
    return safe_risk, {
        "safe_risk_mean": safe_stats["mean"],
        "safe_risk_max": safe_stats["max"],
        "safe_risk_p95": safe_stats["p95"],
        "mu_mean": mu_stats["mean"],
        "mu_max": mu_stats["max"],
        "sigma_mean": sigma_stats["mean"],
        "sigma_max": sigma_stats["max"],
    }


def score_candidate_trajectories(
    *,
    safe_risk_map: torch.Tensor,
    trajectories: torch.Tensor,
    start_xy: torch.Tensor,
    goal_xy: torch.Tensor,
    delta: float,
    goal_weight: float,
    progress_weight: float,
    backtrack_penalty: float,
) -> dict[str, Any]:
    trajectory_risks = evaluate_trajectory_risk(safe_risk_map, trajectories).detach().cpu().float()
    trajectories = trajectories.detach().cpu().float()
    goal = goal_xy.detach().cpu().float()
    start = start_xy.detach().cpu().float()
    final_distances = torch.linalg.norm(trajectories[:, -1] - goal.view(1, 2), dim=1)
    start_distance = torch.linalg.norm(start - goal).clamp_min(1e-6)
    progress_cost = final_distances / start_distance
    backtrack_mask = final_distances > start_distance
    scores = (
        trajectory_risks
        + goal_weight * final_distances
        + progress_weight * progress_cost
        + backtrack_penalty * backtrack_mask.to(dtype=trajectory_risks.dtype)
    )
    feasible_mask = trajectory_risks < delta
    num_feasible = int(feasible_mask.sum().item())
    if num_feasible > 0:
        feasible_indices = torch.nonzero(feasible_mask, as_tuple=False).flatten()
        local_best = int(torch.argmin(scores[feasible_indices]).item())
        best_idx = int(feasible_indices[local_best].item())
    else:
        best_idx = int(torch.argmin(trajectory_risks).item())

    return {
        "best_idx": best_idx,
        "trajectory_risks": trajectory_risks,
        "final_distances": final_distances,
        "scores": scores,
        "feasible_mask": feasible_mask,
        "num_feasible": num_feasible,
    }


def run_diagnostic_episode(
    *,
    method: str,
    config: dict[str, Any],
    hard: dict[str, Any],
    device: torch.device,
    seed: int,
    episode: int,
    model: torch.nn.Module | None,
    record_trace: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env = build_hard_env(config, hard, seed=seed)
    observation = env.reset(seed=seed)

    horizon = int(config.get("horizon", 5))
    map_size = int(config.get("env", {}).get("map_size", config.get("image_size", 64)))
    num_trajectories = int(config.get("num_trajectories", 64))
    delta = float(config.get("risk_threshold_delta", 1.4))
    goal_weight = float(config.get("goal_weight", 0.3))
    progress_weight = float(config.get("progress_weight", 0.2))
    backtrack_penalty = float(config.get("backtrack_penalty", 1.0))
    noise_scale = float(config.get("trajectory_noise_scale", 5.0))
    max_step_size = float(config.get("env", {}).get("max_step_size", 2.0))

    rows: list[dict[str, Any]] = []
    path = [observation["robot_xy"].clone()]
    cumulative_risk = 0.0
    done = False
    step = 0
    info: dict[str, Any] = {
        "success": False,
        "collision": False,
        "timeout": False,
        "failure_reason": "none",
        "risk_at_robot": 0.0,
        "timestep": 0,
    }
    last_safe_risk_map: torch.Tensor | None = None

    initial_maps = {
        "risk_map": env.risk_map.clone(),
        "fire_map": env.fire_map.clone(),
        "smoke_map": env.smoke_map.clone(),
        "obstacle_map": env.obstacle_map.clone(),
    }

    while not done:
        robot_xy = observation["robot_xy"].clone()
        goal_xy = observation["goal_xy"].clone()
        distance_to_goal = float(torch.linalg.norm(robot_xy - goal_xy).item())
        step_seed = int(seed * 1009 + step * 9176 + 17)
        set_deterministic_seed(step_seed)
        planner_risk, risk_stats = build_diagnostic_risk(method, observation, model, config, device)
        last_safe_risk_map = planner_risk.clone()

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
        candidate_info = score_candidate_trajectories(
            safe_risk_map=planner_risk,
            trajectories=trajectories,
            start_xy=robot_xy,
            goal_xy=goal_xy,
            delta=delta,
            goal_weight=goal_weight,
            progress_weight=progress_weight,
            backtrack_penalty=backtrack_penalty,
        )
        best_idx = int(candidate_info["best_idx"])
        best_traj = trajectories[best_idx]
        selected_action = best_traj[0]

        observation, done, info = env.step(selected_action)
        path.append(observation["robot_xy"].clone())
        cumulative_risk += float(info["risk_at_robot"])

        trajectory_risks = candidate_info["trajectory_risks"]
        final_distances = candidate_info["final_distances"]
        scores = candidate_info["scores"]
        num_feasible = int(candidate_info["num_feasible"])
        failure_reason = str(info.get("failure_reason", "none")) if done else "none"
        if done and bool(info.get("success", False)):
            failure_reason = "reached_goal"

        rows.append(
            {
                "episode": episode,
                "seed": seed,
                "method": method,
                "step": step,
                "robot_x": float(robot_xy[0].item()),
                "robot_y": float(robot_xy[1].item()),
                "goal_x": float(goal_xy[0].item()),
                "goal_y": float(goal_xy[1].item()),
                "distance_to_goal": distance_to_goal,
                "selected_action_x": float(selected_action[0].item()),
                "selected_action_y": float(selected_action[1].item()),
                "selected_traj_risk": float(trajectory_risks[best_idx].item()),
                "selected_final_distance": float(final_distances[best_idx].item()),
                "selected_score": float(scores[best_idx].item()),
                "num_candidates": int(trajectories.shape[0]),
                "num_feasible": num_feasible,
                "feasible_ratio": num_feasible / max(1, int(trajectories.shape[0])),
                "min_candidate_risk": float(trajectory_risks.min().item()),
                "mean_candidate_risk": float(trajectory_risks.mean().item()),
                "max_candidate_risk": float(trajectory_risks.max().item()),
                "min_goal_distance": float(final_distances.min().item()),
                "mean_goal_distance": float(final_distances.mean().item()),
                "safe_risk_mean": risk_stats["safe_risk_mean"],
                "safe_risk_max": risk_stats["safe_risk_max"],
                "safe_risk_p95": risk_stats["safe_risk_p95"],
                "mu_mean": risk_stats["mu_mean"],
                "mu_max": risk_stats["mu_max"],
                "sigma_mean": risk_stats["sigma_mean"],
                "sigma_max": risk_stats["sigma_max"],
                "collision": int(bool(info.get("collision", False))),
                "success": int(bool(info.get("success", False))),
                "failure_reason": failure_reason,
            }
        )
        step += 1

    metrics: dict[str, Any] = {
        "method": method,
        "episode": episode,
        "seed": seed,
        "success": bool(info.get("success", False)),
        "collision": bool(info.get("collision", False)),
        "timeout": bool(info.get("timeout", False)),
        "failure_reason": "reached_goal" if bool(info.get("success", False)) else str(info.get("failure_reason", "none")),
        "cumulative_risk": cumulative_risk,
        "path_length": float(info.get("path_length", compute_path_length(path))),
        "steps": int(info.get("timestep", len(path) - 1)),
    }
    if record_trace:
        metrics.update(
            {
                "path": torch.stack(path, dim=0),
                "start_xy": env.start_xy.clone(),
                "goal_xy": env.goal_xy.clone(),
                "initial_maps": initial_maps,
                "final_risk_map": env.risk_map.clone(),
                "final_fire_map": env.fire_map.clone(),
                "final_smoke_map": env.smoke_map.clone(),
                "final_obstacle_map": env.obstacle_map.clone(),
                "last_safe_risk_map": last_safe_risk_map,
            }
        )
    return rows, metrics


def validate_methods(methods: list[str]) -> list[str]:
    unique_methods = list(dict.fromkeys(methods))
    invalid = [method for method in unique_methods if method not in STAGE5_METHODS]
    if invalid:
        raise ValueError(f"Unknown methods {invalid}. Valid methods: {STAGE5_METHODS}")
    return unique_methods


def load_stage5_2_config(config_path: Path, hard_config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    base_config = load_config(config_path)
    hard = load_hard_config(hard_config_path)
    return apply_hard_eval_config(base_config, hard), hard


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError(f"--episodes must be positive, got {args.episodes}")
    methods = validate_methods(args.methods)
    config, hard = load_stage5_2_config(args.config, args.hard_config)
    base_seed = int(config.get("data", {}).get("seed", config.get("seed", 42)))
    seeds = [base_seed + idx for idx in range(args.episodes)]
    device = resolve_device(args.device)
    model = build_loaded_model(config, args.checkpoint, device) if needs_model(methods) else None

    all_rows: list[dict[str, Any]] = []
    print("Stage-5.2 diagnostic methods:", " ".join(methods))
    print("Seed list:", seeds)
    for method in methods:
        for episode, seed in enumerate(seeds):
            rows, metrics = run_diagnostic_episode(
                method=method,
                config=config,
                hard=hard,
                device=device,
                seed=seed,
                episode=episode,
                model=model if method in MODEL_METHODS else None,
            )
            all_rows.extend(rows)
            print(
                f"method={method} episode={episode + 1}/{args.episodes} seed={seed} "
                f"success={metrics['success']} collision={metrics['collision']} "
                f"failure_reason={metrics['failure_reason']} steps={metrics['steps']} "
                f"risk={metrics['cumulative_risk']:.4f}"
            )

    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DIAGNOSTIC_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved Stage-5.2 hard planner diagnostics to: {output}")


if __name__ == "__main__":
    main()
