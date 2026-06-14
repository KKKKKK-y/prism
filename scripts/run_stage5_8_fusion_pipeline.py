from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = PROJECT_ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from prism.scripts.sweep_fusion_planner import run_sweep, sort_rows


CHECKPOINTS = [
    {
        "checkpoint_name": "unweighted_hard_b",
        "config": Path("configs/toy_train_hard_level_b.yaml"),
        "checkpoint": Path("outputs/checkpoints_toy_hard_b/best.pt"),
    },
    {
        "checkpoint_name": "weighted_w20",
        "config": Path("configs/toy_train_hard_weighted_w20.yaml"),
        "checkpoint": Path("outputs/checkpoints_sweep/toy_train_hard_weighted_w20/best.pt"),
    },
    {
        "checkpoint_name": "weighted_w50",
        "config": Path("configs/toy_train_hard_weighted_w50.yaml"),
        "checkpoint": Path("outputs/checkpoints_sweep/toy_train_hard_weighted_w50/best.pt"),
    },
]

ALL_FIELDS = [
    "checkpoint_name",
    "config",
    "checkpoint",
    "mode",
    "alpha",
    "scale",
    "success_rate",
    "collision_rate",
    "timeout_rate",
    "avg_cumulative_risk",
    "avg_path_length",
    "avg_steps",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-5.8 fusion sweep for all available checkpoints.")
    parser.add_argument("--hard-config", type=Path, default=Path("configs/toy_eval_hard.yaml"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("outputs/results/stage5_8_fusion_all_checkpoints.csv"))
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    return parser.parse_args()


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def sort_all_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            float(row["collision_rate"]),
            -float(row["success_rate"]),
            float(row["avg_cumulative_risk"]),
            float(row["avg_path_length"]),
        ),
    )


def print_top(rows: list[dict[str, Any]], limit: int = 10) -> None:
    print("\nStage-5.8 all-checkpoint top 10:")
    for idx, row in enumerate(rows[:limit], start=1):
        print(
            f"{idx}. checkpoint={row['checkpoint_name']} mode={row['mode']} "
            f"alpha={row['alpha']} scale={row['scale']} "
            f"success={float(row['success_rate']):.6f} collision={float(row['collision_rate']):.6f} "
            f"timeout={float(row['timeout_rate']):.6f} risk={float(row['avg_cumulative_risk']):.6f} "
            f"path={float(row['avg_path_length']):.6f} steps={float(row['avg_steps']):.6f}"
        )


def main() -> None:
    args = parse_args()
    all_rows: list[dict[str, Any]] = []
    for item in CHECKPOINTS:
        config = repo_path(item["config"])
        checkpoint = repo_path(item["checkpoint"])
        if not config.exists():
            print(f"Warning: missing config, skipping {item['checkpoint_name']}: {item['config']}")
            continue
        if not checkpoint.exists():
            print(f"Warning: missing checkpoint, skipping {item['checkpoint_name']}: {item['checkpoint']}")
            continue
        sweep_args = argparse.Namespace(
            config=item["config"],
            hard_config=args.hard_config,
            checkpoint=item["checkpoint"],
            episodes=args.episodes,
            output=Path("outputs/results") / f"stage5_8_fusion_sweep_{item['checkpoint_name']}.csv",
            device=args.device,
        )
        print(f"\nRunning Stage-5.8 fusion sweep for {item['checkpoint_name']}")
        rows = run_sweep(sweep_args)
        checkpoint_rows = []
        for row in rows:
            row = dict(row)
            row.pop("episodes", None)
            checkpoint_rows.append(
                {
                    "checkpoint_name": item["checkpoint_name"],
                    "config": str(item["config"]),
                    "checkpoint": str(item["checkpoint"]),
                    **row,
                }
            )
        sweep_args.output.parent.mkdir(parents=True, exist_ok=True)
        with sweep_args.output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ALL_FIELDS)
            writer.writeheader()
            writer.writerows(checkpoint_rows)
        all_rows.extend(checkpoint_rows)

    sorted_rows = sort_all_rows(all_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_FIELDS)
        writer.writeheader()
        writer.writerows(sorted_rows)
    print_top(sorted_rows)
    print(f"Saved Stage-5.8 all-checkpoint fusion results to: {args.output}")


if __name__ == "__main__":
    main()
