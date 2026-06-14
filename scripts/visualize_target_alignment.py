from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = PROJECT_ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from prism.scripts.diagnose_hard_planner import load_stage5_2_config
from prism.scripts.test_target_rollout_alignment import generate_episode_trace, mae, true_rollout_from_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize hard target-vs-rollout alignment.")
    parser.add_argument("--config", type=Path, default=Path("configs/toy_train_hard_level_b.yaml"))
    parser.add_argument("--hard-config", type=Path, default=Path("configs/toy_eval_hard.yaml"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/visualizations/stage5_5_target_alignment_seed0.png"),
    )
    parser.add_argument(
        "--shift-output",
        type=Path,
        default=Path("outputs/visualizations/stage5_5_target_horizon_shift_seed0.png"),
    )
    return parser.parse_args()


def _map(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach().cpu().float()
    while value.ndim > 2:
        value = value[0]
    return value


def save_alignment_plot(current: torch.Tensor, target: torch.Tensor, true_rollout: torch.Tensor, seed: int, output: Path) -> None:
    horizon = int(target.shape[0])
    rows = [
        ("current risk", torch.stack([current for _ in range(horizon)], dim=0), 0.0, 1.0, "inferno"),
        ("dataset target", target, 0.0, 1.0, "inferno"),
        ("true rollout risk", true_rollout[:horizon], 0.0, 1.0, "inferno"),
        ("abs difference", (target - true_rollout[:horizon]).abs(), 0.0, max(0.01, float((target - true_rollout[:horizon]).abs().max())), "magma"),
    ]
    fig, axes = plt.subplots(len(rows), horizon, figsize=(3.0 * horizon, 3.0 * len(rows)), constrained_layout=True)
    for row_idx, (label, maps, vmin, vmax, cmap) in enumerate(rows):
        for horizon_idx in range(horizon):
            ax = axes[row_idx, horizon_idx]
            ax.imshow(_map(maps[horizon_idx]).numpy(), cmap=cmap, origin="upper", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if horizon_idx == 0:
                ax.set_ylabel(label, fontsize=10)
            if row_idx == 0:
                h_mae = mae(target[horizon_idx], true_rollout[horizon_idx])
                ax.set_title(f"h={horizon_idx + 1}\nMAE={h_mae:.6f}", fontsize=9)
    fig.suptitle(f"Stage-5.5 target alignment, seed={seed}", fontsize=14)
    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def save_shift_plot(current: torch.Tensor, target: torch.Tensor, true_extended: torch.Tensor, seed: int, output: Path) -> None:
    horizon = int(target.shape[0])
    minus = torch.stack([current if h == 0 else true_extended[h - 1] for h in range(horizon)], dim=0)
    zero = true_extended[:horizon]
    plus = true_extended[1 : horizon + 1]
    rows = [
        ("target", target, 0.0, 1.0, "inferno"),
        ("abs diff shift -1", (target - minus).abs(), 0.0, max(0.01, float((target - minus).abs().max())), "magma"),
        ("abs diff shift 0", (target - zero).abs(), 0.0, max(0.01, float((target - zero).abs().max())), "magma"),
        ("abs diff shift +1", (target - plus).abs(), 0.0, max(0.01, float((target - plus).abs().max())), "magma"),
    ]
    fig, axes = plt.subplots(len(rows), horizon, figsize=(3.0 * horizon, 3.0 * len(rows)), constrained_layout=True)
    for row_idx, (label, maps, vmin, vmax, cmap) in enumerate(rows):
        for horizon_idx in range(horizon):
            ax = axes[row_idx, horizon_idx]
            ax.imshow(_map(maps[horizon_idx]).numpy(), cmap=cmap, origin="upper", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if horizon_idx == 0:
                ax.set_ylabel(label, fontsize=10)
            if row_idx == 0:
                ax.set_title(f"h={horizon_idx + 1}", fontsize=9)
    fig.suptitle(f"Stage-5.5 horizon shift check, seed={seed}", fontsize=14)
    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config, hard = load_stage5_2_config(args.config, args.hard_config)
    obs_window = int(config.get("obs_window", 4))
    horizon = int(config.get("horizon", 5))
    generator = torch.Generator().manual_seed(args.seed)
    risk_maps, states = generate_episode_trace(config=config, hard=hard, seed=args.seed, generator=generator)
    sample_index = obs_window - 1
    current = risk_maps[sample_index]
    target = torch.stack(risk_maps[sample_index + 1 : sample_index + horizon + 1], dim=0)
    true_extended = true_rollout_from_state(states[sample_index], horizon + 1)
    save_alignment_plot(current, target, true_extended, args.seed, args.output)
    save_shift_plot(current, target, true_extended, args.seed, args.shift_output)
    print(f"Saved target alignment visualization to: {args.output}")
    print(f"Saved target horizon shift visualization to: {args.shift_output}")


if __name__ == "__main__":
    main()
