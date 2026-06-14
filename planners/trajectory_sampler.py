from __future__ import annotations

import math

import numpy as np
import torch


def _as_xy_tensor(value: torch.Tensor | np.ndarray | list[float] | tuple[float, float]) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32)


def sample_candidate_trajectories(
    start_xy: torch.Tensor | np.ndarray | list[float] | tuple[float, float],
    goal_xy: torch.Tensor | np.ndarray | list[float] | tuple[float, float],
    num_trajectories: int = 64,
    horizon: int = 5,
    map_size: int = 64,
    noise_scale: float = 4.0,
    max_step_size: float | None = None,
) -> torch.Tensor:
    """
    Sample noisy candidate trajectories.

    Trajectory coordinates are always [x, y].
    Image/map indexing later uses [y, x], where y is row and x is column.

    When max_step_size is provided, every adjacent waypoint is clipped to that
    motion budget so the sampled rollout matches ToyFireEnv.step dynamics.

    Returns:
        trajectories: [num_trajectories, horizon, 2]
    """
    if num_trajectories <= 0:
        raise ValueError(f"num_trajectories must be positive, got {num_trajectories}")
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if map_size <= 1:
        raise ValueError(f"map_size must be greater than 1, got {map_size}")

    start = _as_xy_tensor(start_xy)
    goal = _as_xy_tensor(goal_xy).to(device=start.device)
    if tuple(start.shape) != (2,) or tuple(goal.shape) != (2,):
        raise ValueError(f"start_xy and goal_xy must both be shape [2], got {tuple(start.shape)} and {tuple(goal.shape)}")

    start = start.clamp(0.0, float(map_size - 1))
    goal = goal.clamp(0.0, float(map_size - 1))

    if max_step_size is not None:
        if max_step_size <= 0.0:
            raise ValueError(f"max_step_size must be positive when provided, got {max_step_size}")
        trajectories = torch.empty(num_trajectories, horizon, 2, device=start.device, dtype=start.dtype)
        noise_weights = torch.sin(torch.linspace(1.0 / horizon, 1.0, horizon, device=start.device, dtype=start.dtype) * math.pi)
        noise_weights[-1] = 0.15
        for traj_idx in range(num_trajectories):
            current = start.clone()
            for step_idx in range(horizon):
                to_goal = goal - current
                distance_to_goal = torch.linalg.norm(to_goal).clamp_min(1e-8)
                base_next = current + to_goal / distance_to_goal * torch.minimum(
                    distance_to_goal,
                    torch.as_tensor(max_step_size, dtype=start.dtype, device=start.device),
                )
                if traj_idx == 0:
                    target = base_next
                else:
                    target = base_next + torch.randn(2, device=start.device, dtype=start.dtype) * noise_scale * noise_weights[step_idx]
                target = target.clamp(0.0, float(map_size - 1))
                delta = target - current
                delta_norm = torch.linalg.norm(delta)
                if delta_norm > max_step_size:
                    target = current + delta / delta_norm * max_step_size
                current = target.clamp(0.0, float(map_size - 1))
                trajectories[traj_idx, step_idx] = current
        return trajectories

    steps = torch.linspace(1.0 / horizon, 1.0, horizon, device=start.device, dtype=start.dtype)
    base_path = start.view(1, 2) + steps.view(horizon, 1) * (goal - start).view(1, 2)
    trajectories = base_path.unsqueeze(0).repeat(num_trajectories, 1, 1)

    noise = torch.randn(num_trajectories, horizon, 2, device=start.device, dtype=start.dtype) * noise_scale
    noise_weight = torch.sin(steps * math.pi).view(1, horizon, 1)
    noise_weight[:, -1] = 0.15
    trajectories = trajectories + noise * noise_weight

    trajectories[0] = base_path
    trajectories[0, -1] = goal
    trajectories = trajectories.clamp(0.0, float(map_size - 1))
    return trajectories
