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
) -> torch.Tensor:
    """
    Sample noisy candidate trajectories.

    Trajectory coordinates are always [x, y].
    Image/map indexing later uses [y, x], where y is row and x is column.

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
