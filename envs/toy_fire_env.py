from __future__ import annotations

from collections import deque
from typing import Any

import torch
import torch.nn.functional as F


class ToyFireEnv:
    """
    Minimal 2D dynamic fire environment for PRISM closed-loop evaluation.

    Coordinate convention:
    - robot_xy, goal_xy, actions and trajectories use [x, y].
    - map/image indexing uses [y, x], where y is row and x is column.
    """

    def __init__(
        self,
        map_size: int = 64,
        obs_window: int = 4,
        channels: int = 5,
        max_steps: int = 100,
        num_fire_sources: int = 2,
        obstacle_density: float = 0.05,
        fire_spread_rate: float = 0.08,
        smoke_spread_rate: float = 0.12,
        risk_threshold_collision: float = 0.85,
        goal_radius: float = 3.0,
        max_step_size: float = 2.0,
        start_xy: tuple[float, float] = (5.0, 58.0),
        goal_xy: tuple[float, float] = (56.0, 8.0),
        seed: int | None = None,
    ) -> None:
        self.map_size = map_size
        self.obs_window = obs_window
        self.channels = channels
        self.max_steps = max_steps
        self.num_fire_sources = num_fire_sources
        self.obstacle_density = obstacle_density
        self.fire_spread_rate = fire_spread_rate
        self.smoke_spread_rate = smoke_spread_rate
        self.risk_threshold_collision = risk_threshold_collision
        self.goal_radius = goal_radius
        self.max_step_size = max_step_size
        self.start_xy = torch.tensor(start_xy, dtype=torch.float32)
        self.goal_xy_default = torch.tensor(goal_xy, dtype=torch.float32)
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

        self.fire_sources: list[tuple[int, int]] = []
        self.history: deque[torch.Tensor] = deque(maxlen=obs_window)
        self.timestep = 0
        self.fire_map = torch.zeros(map_size, map_size)
        self.smoke_map = torch.zeros(map_size, map_size)
        self.obstacle_map = torch.zeros(map_size, map_size)
        self.risk_map = torch.zeros(map_size, map_size)
        self.robot_xy = self.start_xy.clone()
        self.goal_xy = self.goal_xy_default.clone()
        self.total_path_length = 0.0

    def reset(self, seed: int | None = None) -> dict[str, torch.Tensor]:
        if seed is not None:
            self.generator.manual_seed(seed)

        self.timestep = 0
        self.robot_xy = self.start_xy.clone()
        self.goal_xy = self.goal_xy_default.clone()
        self.total_path_length = 0.0
        self.fire_map.zero_()
        self.smoke_map.zero_()
        self.obstacle_map.zero_()
        self.risk_map.zero_()
        self.history.clear()
        self.fire_sources = []

        self._sample_obstacles()
        self._sample_fire_sources()
        self._update_risk_map()
        self._append_history()
        return self._get_observation()

    def step(self, action_xy: torch.Tensor | list[float] | tuple[float, float]) -> tuple[dict[str, torch.Tensor], bool, dict[str, Any]]:
        action = torch.as_tensor(action_xy, dtype=torch.float32).flatten()
        if tuple(action.shape) != (2,):
            raise ValueError(f"action_xy must be shape [2], got {tuple(action.shape)}")

        prev_robot_xy = self.robot_xy.clone()
        direction = action - self.robot_xy
        distance = torch.linalg.norm(direction).item()
        if distance > self.max_step_size and distance > 1e-8:
            self.robot_xy = self.robot_xy + direction / distance * self.max_step_size
        else:
            self.robot_xy = action
        self.robot_xy = self.robot_xy.clamp(0.0, float(self.map_size - 1))
        action_distance = float(torch.linalg.norm(self.robot_xy - prev_robot_xy).item())
        self.total_path_length += action_distance
        self.timestep += 1
        self.update_dynamics()
        self._append_history()

        robot_x = int(round(float(self.robot_xy[0].item())))
        robot_y = int(round(float(self.robot_xy[1].item())))
        robot_x = max(0, min(self.map_size - 1, robot_x))
        robot_y = max(0, min(self.map_size - 1, robot_y))

        # robot_xy is [x, y], map indexing is [y, x].
        obstacle_collision = bool(self.obstacle_map[robot_y, robot_x].item() > 0.5)
        risk_at_robot = float(self.risk_map[robot_y, robot_x].item())
        high_risk_collision = risk_at_robot >= self.risk_threshold_collision
        collision = obstacle_collision or high_risk_collision
        distance_to_goal = float(torch.linalg.norm(self.robot_xy - self.goal_xy).item())
        success = distance_to_goal <= self.goal_radius
        timeout = self.timestep >= self.max_steps
        done = bool(success or collision or timeout)
        failure_reason = "none"
        if obstacle_collision:
            failure_reason = "collision_obstacle"
        elif high_risk_collision:
            failure_reason = "collision_high_risk"
        elif timeout:
            failure_reason = "timeout"

        info = {
            "success": bool(success),
            "collision": bool(collision),
            "obstacle_collision": obstacle_collision,
            "high_risk_collision": bool(high_risk_collision),
            "timeout": bool(timeout),
            "failure_reason": failure_reason,
            "risk_at_robot": risk_at_robot,
            "distance_to_goal": distance_to_goal,
            "action_distance": action_distance,
            "path_length": self.total_path_length,
            "timestep": self.timestep,
        }
        return self._get_observation(), done, info

    def update_dynamics(self) -> None:
        self._spread_fire()
        self._spread_smoke()
        self._update_risk_map()

    def _sample_obstacles(self) -> None:
        random_map = torch.rand(self.map_size, self.map_size, generator=self.generator)
        self.obstacle_map = (random_map < self.obstacle_density).float()
        self._clear_disk(self.obstacle_map, self.start_xy, radius=5.0)
        self._clear_disk(self.obstacle_map, self.goal_xy, radius=5.0)
        self._clear_corridor(self.obstacle_map, self.start_xy, self.goal_xy, radius=2.5)

    def _sample_fire_sources(self) -> None:
        count = max(1, min(3, self.num_fire_sources))
        attempts = 0
        while len(self.fire_sources) < count and attempts < 200:
            attempts += 1
            x = int(torch.randint(12, self.map_size - 12, (1,), generator=self.generator).item())
            y = int(torch.randint(12, self.map_size - 12, (1,), generator=self.generator).item())
            xy = torch.tensor([float(x), float(y)])
            if torch.linalg.norm(xy - self.robot_xy).item() < 12.0:
                continue
            if torch.linalg.norm(xy - self.goal_xy).item() < 10.0:
                continue
            if self.obstacle_map[y, x].item() > 0.5:
                continue
            self.fire_sources.append((x, y))
            self._add_gaussian_blob(self.fire_map, xy, amplitude=1.0, sigma=2.5)

        if not self.fire_sources:
            fallback = torch.tensor([float(self.map_size // 2), float(self.map_size // 2)])
            self.fire_sources.append((int(fallback[0].item()), int(fallback[1].item())))
            self._add_gaussian_blob(self.fire_map, fallback, amplitude=1.0, sigma=2.5)

        self.fire_map.clamp_(0.0, 1.0)

    def _spread_fire(self) -> None:
        kernel = torch.tensor(
            [[0.05, 0.10, 0.05], [0.10, 0.40, 0.10], [0.05, 0.10, 0.05]],
            dtype=self.fire_map.dtype,
        ).view(1, 1, 3, 3)
        fire = self.fire_map.view(1, 1, self.map_size, self.map_size)
        spread = F.conv2d(fire, kernel, padding=1).squeeze(0).squeeze(0)
        self.fire_map = (self.fire_map + self.fire_spread_rate * spread).clamp(0.0, 1.0)
        self.fire_map = self.fire_map * (1.0 - 0.35 * self.obstacle_map)
        for x, y in self.fire_sources:
            self.fire_map[y, x] = 1.0

    def _spread_smoke(self) -> None:
        kernel = torch.full((1, 1, 3, 3), 1.0 / 9.0, dtype=self.smoke_map.dtype)
        smoke = self.smoke_map.view(1, 1, self.map_size, self.map_size)
        diffused = F.conv2d(smoke, kernel, padding=1).squeeze(0).squeeze(0)
        self.smoke_map = (
            0.90 * self.smoke_map
            + self.smoke_spread_rate * diffused
            + 0.35 * self.smoke_spread_rate * self.fire_map
        ).clamp(0.0, 1.0)

    def _update_risk_map(self) -> None:
        obstacle_risk = self.obstacle_map
        self.risk_map = (0.5 * self.fire_map + 0.3 * self.smoke_map + 0.2 * obstacle_risk).clamp(0.0, 1.0)

    def _append_history(self) -> None:
        frame = self._make_frame()
        while len(self.history) < self.obs_window - 1:
            self.history.append(frame.clone())
        self.history.append(frame)

    def _get_observation(self) -> dict[str, torch.Tensor]:
        # obs: [k, C, 64, 64]
        obs = torch.stack(list(self.history), dim=0).float()
        return {
            "obs": obs,
            "risk_map": self.risk_map.clone().float(),
            "robot_xy": self.robot_xy.clone().float(),
            "goal_xy": self.goal_xy.clone().float(),
        }

    def _make_frame(self) -> torch.Tensor:
        robot_goal = torch.zeros_like(self.risk_map)
        self._add_gaussian_blob(robot_goal, self.robot_xy, amplitude=1.0, sigma=1.5)
        self._add_gaussian_blob(robot_goal, self.goal_xy, amplitude=0.65, sigma=2.0)
        base = torch.stack(
            [
                self.fire_map,
                self.smoke_map,
                self.obstacle_map,
                self.risk_map,
                robot_goal.clamp(0.0, 1.0),
            ],
            dim=0,
        )
        if self.channels > base.shape[0]:
            padding = torch.zeros(self.channels - base.shape[0], self.map_size, self.map_size)
            base = torch.cat([base, padding], dim=0)
        return base[: self.channels].float()

    def _add_gaussian_blob(self, target: torch.Tensor, xy: torch.Tensor, amplitude: float, sigma: float) -> None:
        yy, xx = torch.meshgrid(
            torch.arange(self.map_size, dtype=torch.float32),
            torch.arange(self.map_size, dtype=torch.float32),
            indexing="ij",
        )
        # xy is [x, y], map indexing/grid is [y, x].
        blob = amplitude * torch.exp(-((xx - xy[0]).pow(2) + (yy - xy[1]).pow(2)) / (2.0 * sigma**2))
        target.copy_(torch.maximum(target, blob.to(target.dtype)))

    def _clear_disk(self, target: torch.Tensor, xy: torch.Tensor, radius: float) -> None:
        yy, xx = torch.meshgrid(
            torch.arange(self.map_size, dtype=torch.float32),
            torch.arange(self.map_size, dtype=torch.float32),
            indexing="ij",
        )
        mask = ((xx - xy[0]).pow(2) + (yy - xy[1]).pow(2)).sqrt() <= radius
        target[mask] = 0.0

    def _clear_corridor(self, target: torch.Tensor, start_xy: torch.Tensor, goal_xy: torch.Tensor, radius: float) -> None:
        steps = max(2, int(torch.linalg.norm(goal_xy - start_xy).item()))
        for alpha in torch.linspace(0.0, 1.0, steps):
            xy = start_xy + alpha * (goal_xy - start_xy)
            self._clear_disk(target, xy, radius=radius)
