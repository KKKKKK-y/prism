from __future__ import annotations

from typing import Any

import torch

from prism.envs.toy_fire_env import ToyFireEnv


class HardToyFireEnv(ToyFireEnv):
    """
    Hard evaluation wrapper for ToyFireEnv.

    The wrapper keeps the original dynamics and observation contract intact, but
    samples denser obstacles and near-path fires to stress future-risk planning.
    """

    def __init__(
        self,
        *args: Any,
        hard_eval: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        hard_eval = dict(hard_eval or {})
        kwargs.setdefault("map_size", int(hard_eval.get("map_size", kwargs.get("map_size", 64))))
        kwargs.setdefault("max_steps", int(hard_eval.get("max_steps", kwargs.get("max_steps", 90))))
        kwargs.setdefault(
            "num_fire_sources",
            int(hard_eval.get("fire_source_count_max", kwargs.get("num_fire_sources", 5))),
        )
        kwargs.setdefault(
            "obstacle_density",
            float(hard_eval.get("obstacle_density", kwargs.get("obstacle_density", 0.12))),
        )
        kwargs.setdefault(
            "fire_spread_rate",
            float(hard_eval.get("fire_spread_rate", kwargs.get("fire_spread_rate", 0.08))),
        )
        kwargs.setdefault(
            "smoke_spread_rate",
            float(hard_eval.get("smoke_spread_rate", kwargs.get("smoke_spread_rate", 0.14))),
        )
        kwargs.setdefault(
            "risk_threshold_collision",
            float(hard_eval.get("risk_threshold_collision", kwargs.get("risk_threshold_collision", 0.70))),
        )
        super().__init__(*args, **kwargs)

        self.hard_eval = hard_eval
        self.fire_source_count_min = int(hard_eval.get("fire_source_count_min", 2))
        self.fire_source_count_max = int(hard_eval.get("fire_source_count_max", 5))
        self.narrow_passage = bool(hard_eval.get("narrow_passage", True))
        self.chokepoint_bands = bool(hard_eval.get("chokepoint_bands", self.narrow_passage))
        self.dynamic_blocking = bool(hard_eval.get("dynamic_blocking", True))
        self.start_goal_min_distance = float(hard_eval.get("start_goal_min_distance", 45.0))
        self.place_fire_near_shortest_path = bool(hard_eval.get("place_fire_near_shortest_path", True))
        self.path_fire_offset_min = float(hard_eval.get("path_fire_offset_min", 3.0))
        self.path_fire_offset_max = float(hard_eval.get("path_fire_offset_max", 8.0))
        self.corridor_radius = float(hard_eval.get("corridor_radius", 2.0))
        self.dynamic_smoke_gain = float(hard_eval.get("dynamic_smoke_gain", 0.025))

    def reset(self, seed: int | None = None) -> dict[str, torch.Tensor]:
        if seed is not None:
            self.generator.manual_seed(seed)
        self._ensure_hard_start_goal_distance()
        return super().reset(seed=None)

    def _ensure_hard_start_goal_distance(self) -> None:
        if torch.linalg.norm(self.goal_xy_default - self.start_xy).item() >= self.start_goal_min_distance:
            return

        margin = 5.0
        candidates = [
            ((margin, float(self.map_size) - margin - 1.0), (float(self.map_size) - margin - 1.0, margin)),
            ((margin, margin), (float(self.map_size) - margin - 1.0, float(self.map_size) - margin - 1.0)),
            ((float(self.map_size) - margin - 1.0, margin), (margin, float(self.map_size) - margin - 1.0)),
            ((float(self.map_size) - margin - 1.0, float(self.map_size) - margin - 1.0), (margin, margin)),
        ]
        valid = []
        for start_xy, goal_xy in candidates:
            start = torch.tensor(start_xy, dtype=torch.float32)
            goal = torch.tensor(goal_xy, dtype=torch.float32)
            if torch.linalg.norm(goal - start).item() >= self.start_goal_min_distance:
                valid.append((start, goal))
        if not valid:
            return
        idx = int(torch.randint(0, len(valid), (1,), generator=self.generator).item())
        self.start_xy = valid[idx][0]
        self.goal_xy_default = valid[idx][1]

    def _sample_obstacles(self) -> None:
        random_map = torch.rand(self.map_size, self.map_size, generator=self.generator)
        self.obstacle_map = (random_map < self.obstacle_density).float()
        self._clear_disk(self.obstacle_map, self.start_xy, radius=5.0)
        self._clear_disk(self.obstacle_map, self.goal_xy, radius=5.0)

        corridor_radius = self.corridor_radius if self.narrow_passage else 3.0
        self._clear_corridor(self.obstacle_map, self.start_xy, self.goal_xy, radius=corridor_radius)
        if self.narrow_passage and self.chokepoint_bands:
            self._add_chokepoint_bands()

    def _add_chokepoint_bands(self) -> None:
        start = self.start_xy
        goal = self.goal_xy
        direction = goal - start
        norm = torch.linalg.norm(direction).clamp_min(1e-6)
        direction_unit = direction / norm
        perpendicular = torch.stack((-direction_unit[1], direction_unit[0])).to(dtype=torch.float32)
        yy, xx = torch.meshgrid(
            torch.arange(self.map_size, dtype=torch.float32),
            torch.arange(self.map_size, dtype=torch.float32),
            indexing="ij",
        )
        points = torch.stack([xx, yy], dim=-1)
        for alpha in (0.35, 0.65):
            center = start + alpha * direction
            relative = points - center.view(1, 1, 2)
            along = (relative * direction_unit.view(1, 1, 2)).sum(dim=-1).abs()
            across = (relative * perpendicular.view(1, 1, 2)).sum(dim=-1).abs()
            band = (along <= 1.0) & (across <= 17.0)
            self.obstacle_map[band] = 1.0
            self._clear_disk(self.obstacle_map, center, radius=3.0)
        self._clear_disk(self.obstacle_map, self.start_xy, radius=5.0)
        self._clear_disk(self.obstacle_map, self.goal_xy, radius=5.0)

    def _sample_fire_sources(self) -> None:
        self.fire_sources = []
        min_count = max(1, self.fire_source_count_min)
        max_count = max(min_count, self.fire_source_count_max)
        count = int(torch.randint(min_count, max_count + 1, (1,), generator=self.generator).item())

        if self.place_fire_near_shortest_path:
            self._sample_path_near_fire_sources(count)
        attempts = 0
        while len(self.fire_sources) < count and attempts < 300:
            attempts += 1
            x = int(torch.randint(10, self.map_size - 10, (1,), generator=self.generator).item())
            y = int(torch.randint(10, self.map_size - 10, (1,), generator=self.generator).item())
            self._try_add_fire_source(torch.tensor([float(x), float(y)], dtype=torch.float32))

        if not self.fire_sources:
            fallback = (self.start_xy + self.goal_xy) * 0.5
            self._try_add_fire_source(fallback)

        self.fire_map.clamp_(0.0, 1.0)

    def _sample_path_near_fire_sources(self, target_count: int) -> None:
        start = self.start_xy
        goal = self.goal_xy
        direction = goal - start
        norm = torch.linalg.norm(direction).clamp_min(1e-6)
        direction_unit = direction / norm
        perpendicular = torch.stack((-direction_unit[1], direction_unit[0])).to(dtype=torch.float32)
        attempts = 0
        while len(self.fire_sources) < target_count and attempts < target_count * 80:
            attempts += 1
            alpha = 0.18 + 0.68 * float(torch.rand((), generator=self.generator).item())
            offset_span = max(0.0, self.path_fire_offset_max - self.path_fire_offset_min)
            offset = self.path_fire_offset_min + offset_span * float(torch.rand((), generator=self.generator).item())
            side = -1.0 if float(torch.rand((), generator=self.generator).item()) < 0.5 else 1.0
            xy = start + alpha * direction + side * offset * perpendicular
            self._try_add_fire_source(xy)

    def _try_add_fire_source(self, xy: torch.Tensor) -> bool:
        xy = xy.to(dtype=torch.float32)
        xy = xy.clamp(2.0, float(self.map_size - 3))
        x = int(round(float(xy[0].item())))
        y = int(round(float(xy[1].item())))
        if torch.linalg.norm(xy - self.robot_xy).item() < 9.0:
            return False
        if torch.linalg.norm(xy - self.goal_xy).item() < 8.0:
            return False
        if self.obstacle_map[y, x].item() > 0.5:
            return False
        if (x, y) in self.fire_sources:
            return False
        self.fire_sources.append((x, y))
        self._add_gaussian_blob(self.fire_map, torch.tensor([float(x), float(y)]), amplitude=1.0, sigma=3.0)
        return True

    def update_dynamics(self) -> None:
        super().update_dynamics()
        if not self.dynamic_blocking:
            return
        for x, y in self.fire_sources:
            source_xy = torch.tensor([float(x), float(y)])
            self._add_gaussian_blob(
                self.smoke_map,
                source_xy,
                amplitude=min(1.0, 0.15 + self.dynamic_smoke_gain * float(self.timestep)),
                sigma=4.0,
            )
        self._update_risk_map()
