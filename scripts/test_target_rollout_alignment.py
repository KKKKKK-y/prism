from __future__ import annotations

import argparse
import copy
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

from prism.scripts.diagnose_hard_planner import load_stage5_2_config
from prism.scripts.diagnose_prediction_calibration import mae, pearson_corr, rmse, tensor_stats
from prism.scripts.evaluate_baselines_hard import build_hard_env
from prism.scripts.generate_toy_dataset import exploratory_action


ALIGNMENT_FIELDS = [
    "episode",
    "sample_index",
    "horizon",
    "target_mean",
    "target_max",
    "target_p95",
    "true_mean",
    "true_max",
    "true_p95",
    "mae",
    "rmse",
    "corr",
    "max_abs_diff",
    "mae_shift_minus1",
    "mae_shift_0",
    "mae_shift_plus1",
    "best_aligned_shift",
    "current_to_target_mae",
    "current_to_target_corr",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check hard target generation against true future rollout.")
    parser.add_argument("--config", type=Path, default=Path("configs/toy_train_hard_level_b.yaml"))
    parser.add_argument("--hard-config", type=Path, default=Path("configs/toy_eval_hard.yaml"))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/results/stage5_5_target_rollout_alignment.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/results/stage5_5_target_rollout_alignment_summary.txt"),
    )
    return parser.parse_args()


def true_rollout_from_state(env: Any, steps: int) -> torch.Tensor:
    rollout_env = copy.deepcopy(env)
    maps: list[torch.Tensor] = []
    for _ in range(steps):
        rollout_env.timestep += 1
        rollout_env.update_dynamics()
        maps.append(rollout_env.risk_map.detach().clone().float())
    return torch.stack(maps, dim=0)


def generate_episode_trace(
    *,
    config: dict[str, Any],
    hard: dict[str, Any],
    seed: int,
    generator: torch.Generator,
) -> tuple[list[torch.Tensor], list[Any]]:
    max_steps = int(config.get("env", {}).get("max_steps", 100))
    env = build_hard_env(config, hard, seed=seed)
    observation = env.reset(seed=seed)
    risk_maps = [observation["risk_map"].detach().clone().float()]
    states = [copy.deepcopy(env)]
    for _ in range(max_steps):
        action_xy = exploratory_action(env, generator)
        observation, _, _ = env.step(action_xy)
        risk_maps.append(observation["risk_map"].detach().clone().float())
        states.append(copy.deepcopy(env))
    return risk_maps, states


def shift_reference(current: torch.Tensor, true_extended: torch.Tensor, horizon_idx: int, shift: int) -> torch.Tensor:
    if shift == -1:
        if horizon_idx == 0:
            return current
        return true_extended[horizon_idx - 1]
    if shift == 0:
        return true_extended[horizon_idx]
    if shift == 1:
        return true_extended[horizon_idx + 1]
    raise ValueError(f"Unsupported shift: {shift}")


def row_for_horizon(
    *,
    episode: int,
    sample_index: int,
    horizon_idx: int,
    current: torch.Tensor,
    target: torch.Tensor,
    true_extended: torch.Tensor,
) -> dict[str, Any]:
    true_risk = true_extended[horizon_idx]
    target_stats = tensor_stats(target)
    true_stats = tensor_stats(true_risk)
    shift_maes = {
        -1: mae(target, shift_reference(current, true_extended, horizon_idx, -1)),
        0: mae(target, shift_reference(current, true_extended, horizon_idx, 0)),
        1: mae(target, shift_reference(current, true_extended, horizon_idx, 1)),
    }
    best_shift = min(shift_maes, key=shift_maes.get)
    return {
        "episode": episode,
        "sample_index": sample_index,
        "horizon": horizon_idx + 1,
        "target_mean": target_stats["mean"],
        "target_max": target_stats["max"],
        "target_p95": target_stats["p95"],
        "true_mean": true_stats["mean"],
        "true_max": true_stats["max"],
        "true_p95": true_stats["p95"],
        "mae": mae(target, true_risk),
        "rmse": rmse(target, true_risk),
        "corr": pearson_corr(target, true_risk),
        "max_abs_diff": float((target - true_risk).abs().max().item()),
        "mae_shift_minus1": shift_maes[-1],
        "mae_shift_0": shift_maes[0],
        "mae_shift_plus1": shift_maes[1],
        "best_aligned_shift": best_shift,
        "current_to_target_mae": mae(current, target),
        "current_to_target_corr": pearson_corr(current, target),
    }


def mean(values: list[float], ignore_nan: bool = True) -> float:
    if ignore_nan:
        values = [value for value in values if not math.isnan(value)]
    if not values:
        return float("nan")
    return sum(values) / len(values)


def write_summary(rows: list[dict[str, Any]], output: Path) -> None:
    lines = ["PRISM Stage-5.5 Target Rollout Alignment Summary", "=================================================", ""]
    lines.append(f"rows: {len(rows)}")
    lines.append(f"overall_mae: {mean([float(row['mae']) for row in rows]):.8f}")
    lines.append(f"overall_rmse: {mean([float(row['rmse']) for row in rows]):.8f}")
    lines.append(f"overall_corr: {mean([float(row['corr']) for row in rows]):.8f}")
    lines.append(f"overall_max_abs_diff: {max(float(row['max_abs_diff']) for row in rows):.8f}")
    lines.append(f"overall_current_to_target_mae: {mean([float(row['current_to_target_mae']) for row in rows]):.8f}")
    lines.append(f"overall_current_to_target_corr: {mean([float(row['current_to_target_corr']) for row in rows]):.8f}")
    shift_counts: dict[int, int] = {-1: 0, 0: 0, 1: 0}
    for row in rows:
        shift_counts[int(row["best_aligned_shift"])] += 1
    lines.append(f"best_shift_counts: minus1={shift_counts[-1]} zero={shift_counts[0]} plus1={shift_counts[1]}")
    lines.append("")
    lines.append("horizon mae rmse corr max_abs_diff current_to_target_mae current_to_target_corr best_shift_mode")
    for horizon in sorted({int(row["horizon"]) for row in rows}):
        h_rows = [row for row in rows if int(row["horizon"]) == horizon]
        h_shift_counts: dict[int, int] = {-1: 0, 0: 0, 1: 0}
        for row in h_rows:
            h_shift_counts[int(row["best_aligned_shift"])] += 1
        shift_mode = max(h_shift_counts, key=h_shift_counts.get)
        lines.append(
            f"{horizon} "
            f"{mean([float(row['mae']) for row in h_rows]):.8f} "
            f"{mean([float(row['rmse']) for row in h_rows]):.8f} "
            f"{mean([float(row['corr']) for row in h_rows]):.8f} "
            f"{max(float(row['max_abs_diff']) for row in h_rows):.8f} "
            f"{mean([float(row['current_to_target_mae']) for row in h_rows]):.8f} "
            f"{mean([float(row['current_to_target_corr']) for row in h_rows]):.8f} "
            f"{shift_mode}"
        )
    lines.append("")
    if max(float(row["max_abs_diff"]) for row in rows) <= 1e-6:
        lines.append("alignment_answer: target exactly matches true rollout for checked samples.")
    else:
        lines.append("alignment_answer: target differs from true rollout for checked samples.")
    if shift_counts[0] >= max(shift_counts[-1], shift_counts[1]):
        lines.append("horizon_shift_answer: no dominant off-by-one shift detected.")
    else:
        lines.append("horizon_shift_answer: possible off-by-one shift detected.")
    if mean([float(row["current_to_target_mae"]) for row in rows]) <= 1e-3:
        lines.append("current_copy_answer: target is almost identical to current risk.")
    else:
        lines.append("current_copy_answer: target is not simply current risk.")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved target rollout alignment summary to: {output}")


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError(f"--episodes must be positive, got {args.episodes}")
    config, hard = load_stage5_2_config(args.config, args.hard_config)
    obs_window = int(config.get("obs_window", 4))
    horizon = int(config.get("horizon", 5))
    base_seed = int(config.get("data", {}).get("seed", config.get("seed", 42)))
    generator = torch.Generator().manual_seed(base_seed)

    rows: list[dict[str, Any]] = []
    for episode in range(args.episodes):
        seed = base_seed + episode
        risk_maps, states = generate_episode_trace(config=config, hard=hard, seed=seed, generator=generator)
        sample_index = obs_window - 1
        current = risk_maps[sample_index]
        target = torch.stack(risk_maps[sample_index + 1 : sample_index + horizon + 1], dim=0)
        true_extended = true_rollout_from_state(states[sample_index], horizon + 1)
        for horizon_idx in range(horizon):
            rows.append(
                row_for_horizon(
                    episode=episode,
                    sample_index=sample_index,
                    horizon_idx=horizon_idx,
                    current=current,
                    target=target[horizon_idx],
                    true_extended=true_extended,
                )
            )
        print(f"episode={episode + 1}/{args.episodes} seed={seed} sample_index={sample_index}")

    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ALIGNMENT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved target rollout alignment CSV to: {output}")
    write_summary(rows, args.summary_output.expanduser())


if __name__ == "__main__":
    main()
