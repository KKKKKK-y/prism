from __future__ import annotations

import torch


def _normalize_safe_risk_map(safe_risk_map: torch.Tensor) -> torch.Tensor:
    if safe_risk_map.ndim == 4:
        if safe_risk_map.shape[1] != 1:
            raise ValueError(f"safe_risk_map channel dimension must be 1, got {safe_risk_map.shape[1]}")
        return safe_risk_map[:, 0]
    if safe_risk_map.ndim == 3:
        return safe_risk_map
    raise ValueError(f"safe_risk_map must be [H,1,64,64] or [H,64,64], got {tuple(safe_risk_map.shape)}")


def evaluate_trajectory_risk(safe_risk_map: torch.Tensor, trajectories: torch.Tensor) -> torch.Tensor:
    """
    Sum safe-risk values along each trajectory.

    safe_risk_map: [H, 1, 64, 64] or [H, 64, 64]
    trajectories: [N, H, 2], coordinates are [x, y]

    Map lookup converts trajectory [x, y] to image indexing [y, x].
    Returns:
        trajectory_risks: [N]
    """
    risk_map = _normalize_safe_risk_map(safe_risk_map)
    if trajectories.ndim != 3 or trajectories.shape[-1] != 2:
        raise ValueError(f"trajectories must be [N,H,2], got {tuple(trajectories.shape)}")

    device = risk_map.device
    trajectories = trajectories.to(device=device, dtype=torch.float32)
    map_horizon, height, width = risk_map.shape
    num_trajectories, traj_horizon, _ = trajectories.shape
    if traj_horizon != map_horizon:
        raise ValueError(f"Trajectory horizon {traj_horizon} does not match safe_risk_map horizon {map_horizon}")

    xy = trajectories.round().long()
    xs = xy[..., 0].clamp(0, width - 1)
    ys = xy[..., 1].clamp(0, height - 1)

    horizon_idx = torch.arange(map_horizon, device=device).view(1, map_horizon).expand(num_trajectories, map_horizon)
    risks = risk_map[horizon_idx, ys, xs]
    trajectory_risks = risks.sum(dim=1)
    return trajectory_risks


def select_safe_trajectory(
    safe_risk_map: torch.Tensor,
    trajectories: torch.Tensor,
    goal_xy: torch.Tensor | list[float] | tuple[float, float],
    start_xy: torch.Tensor | list[float] | tuple[float, float] | None = None,
    delta: float = 0.8,
    goal_weight: float = 0.1,
    progress_weight: float = 0.2,
    backtrack_penalty: float = 1.0,
) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor, dict[str, float | int | bool]]:
    """
    Select a trajectory under trajectory-level safe-risk constraint.

    trajectories: [N, H, 2], coordinates are [x, y]
    feasible_mask: [N], true where R_safe(tau) < delta
    """
    trajectory_risks = evaluate_trajectory_risk(safe_risk_map, trajectories)
    device = trajectory_risks.device
    trajectories = trajectories.to(device=device, dtype=torch.float32)
    goal = torch.as_tensor(goal_xy, dtype=torch.float32, device=device)
    if tuple(goal.shape) != (2,):
        raise ValueError(f"goal_xy must be shape [2], got {tuple(goal.shape)}")
    if start_xy is None:
        start = trajectories[:, 0].mean(dim=0)
    else:
        start = torch.as_tensor(start_xy, dtype=torch.float32, device=device)
    if tuple(start.shape) != (2,):
        raise ValueError(f"start_xy must be shape [2], got {tuple(start.shape)}")

    feasible_mask = trajectory_risks < delta
    num_feasible = int(feasible_mask.sum().item())
    min_risk = float(trajectory_risks.min().item())
    final_distance_all = torch.linalg.norm(trajectories[:, -1] - goal.view(1, 2), dim=1)
    start_distance = torch.linalg.norm(start - goal).clamp_min(1e-6)
    progress_cost_all = final_distance_all / start_distance
    backtrack_mask_all = final_distance_all > start_distance
    scores_all = (
        trajectory_risks
        + goal_weight * final_distance_all
        + progress_weight * progress_cost_all
        + backtrack_penalty * backtrack_mask_all.to(dtype=trajectory_risks.dtype)
    )

    if num_feasible > 0:
        feasible_indices = torch.nonzero(feasible_mask, as_tuple=False).flatten()
        scores = scores_all[feasible_indices]
        local_best = int(torch.argmin(scores).item())
        best_idx = int(feasible_indices[local_best].item())
        feasible = True
    else:
        best_idx = int(torch.argmin(trajectory_risks).item())
        feasible = False

    best_traj = trajectories[best_idx]
    best_risk = float(trajectory_risks[best_idx].item())
    best_score = float(scores_all[best_idx].item())
    best_final_distance = float(final_distance_all[best_idx].item())
    info: dict[str, float | int | bool] = {
        "num_feasible": num_feasible,
        "min_risk": min_risk,
        "best_risk": best_risk,
        "best_score": best_score,
        "best_final_distance": best_final_distance,
        "start_distance": float(start_distance.item()),
        "feasible": feasible,
    }
    return best_traj, best_idx, trajectory_risks, feasible_mask, info
