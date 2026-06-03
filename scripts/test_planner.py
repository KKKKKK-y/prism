from __future__ import annotations

import argparse
from pathlib import Path

import torch

from prism.config import load_config
from prism.planners import evaluate_trajectory_risk, sample_candidate_trajectories, select_safe_trajectory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate PRISM Stage-3 trajectory safe-risk constraint.")
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"), help="Path to YAML config.")
    return parser.parse_args()


def build_dummy_safe_risk_map(horizon: int, image_size: int) -> torch.Tensor:
    """
    Build a deterministic dummy safe-risk map.

    safe_risk_map: [H, 1, 64, 64]
    Image/map indexing is [y, x], where y is row and x is column.
    """
    yy, xx = torch.meshgrid(
        torch.arange(image_size, dtype=torch.float32),
        torch.arange(image_size, dtype=torch.float32),
        indexing="ij",
    )
    base = torch.full((image_size, image_size), 0.02, dtype=torch.float32)
    obstacle = 0.30 * torch.exp(-((xx - 31.0).pow(2) + (yy - 34.0).pow(2)) / (2.0 * 8.0**2))
    risk = (base + obstacle).clamp(0.0, 1.0)
    horizon_scale = torch.linspace(0.85, 1.15, horizon).view(horizon, 1, 1)
    return (risk.unsqueeze(0) * horizon_scale).unsqueeze(1).clamp(0.0, 1.0)


def assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{name} contains NaN or Inf values.")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    horizon = int(config.get("horizon", 5))
    image_size = int(config.get("image_size", 64))
    num_trajectories = int(config.get("num_trajectories", 64))
    delta = float(config.get("risk_threshold_delta", 0.8))
    goal_weight = float(config.get("goal_weight", 0.1))
    noise_scale = float(config.get("trajectory_noise_scale", 4.0))

    safe_risk_map = build_dummy_safe_risk_map(horizon, image_size)
    start_xy = torch.tensor([5.0, 58.0])
    goal_xy = torch.tensor([56.0, 8.0])

    # trajectories: [N, H, 2], coordinates are [x, y].
    trajectories = sample_candidate_trajectories(
        start_xy=start_xy,
        goal_xy=goal_xy,
        num_trajectories=num_trajectories,
        horizon=horizon,
        map_size=image_size,
        noise_scale=noise_scale,
    )
    # trajectory_risks: [N], each value sums safe_risk_map[h, y_h, x_h].
    trajectory_risks = evaluate_trajectory_risk(safe_risk_map, trajectories)
    best_traj, _, trajectory_risks, feasible_mask, info = select_safe_trajectory(
        safe_risk_map=safe_risk_map,
        trajectories=trajectories,
        goal_xy=goal_xy,
        start_xy=start_xy,
        delta=delta,
        goal_weight=goal_weight,
        progress_weight=float(config.get("progress_weight", 0.2)),
        backtrack_penalty=float(config.get("backtrack_penalty", 1.0)),
    )

    print(f"trajectories.shape: {tuple(trajectories.shape)}")
    print(f"trajectory_risks.shape: {tuple(trajectory_risks.shape)}")
    print(f"feasible_mask.shape: {tuple(feasible_mask.shape)}")
    print(f"num_feasible: {info['num_feasible']}")
    print(f"min_risk: {info['min_risk']:.6f}")
    print(f"best_risk: {info['best_risk']:.6f}")

    expected_trajectories = (num_trajectories, horizon, 2)
    expected_risks = (num_trajectories,)
    assert tuple(trajectories.shape) == expected_trajectories, f"Unexpected trajectories shape: {tuple(trajectories.shape)}"
    assert tuple(trajectory_risks.shape) == expected_risks, f"Unexpected trajectory_risks shape: {tuple(trajectory_risks.shape)}"
    assert tuple(feasible_mask.shape) == expected_risks, f"Unexpected feasible_mask shape: {tuple(feasible_mask.shape)}"
    assert tuple(best_traj.shape) == (horizon, 2), f"Unexpected best_traj shape: {tuple(best_traj.shape)}"

    assert_finite("trajectories", trajectories)
    assert_finite("trajectory_risks", trajectory_risks)
    assert_finite("best_traj", best_traj)
    print("All PRISM Stage-3 trajectory safe-risk checks passed.")


if __name__ == "__main__":
    main()
