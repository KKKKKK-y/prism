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

from prism.config import load_config
from prism.scripts.evaluate_baselines import build_loaded_model
from prism.scripts.evaluate_baselines_hard import apply_hard_eval_config, load_hard_config
from prism.scripts.evaluate_fusion_planner import FUSION_SUMMARY_FIELDS, evaluate_fusion_setting
from prism.scripts.run_closed_loop import resolve_device


ALPHAS = [0.2, 0.4, 0.6, 0.8]
SCALES = [0.5, 0.75, 1.0, 1.25, 1.5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep Stage-5.8 current/predicted fusion planner parameters.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--hard-config", type=Path, default=Path("configs/toy_eval_hard.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("outputs/results/stage5_8_fusion_sweep.csv"))
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    return parser.parse_args()


def fusion_grid() -> list[tuple[str, float | None, float | None]]:
    grid: list[tuple[str, float | None, float | None]] = [
        ("current_only", None, None),
        ("predicted_only", None, 1.0),
        ("max_fusion", None, None),
    ]
    grid.extend(("alpha_fusion", alpha, None) for alpha in ALPHAS)
    grid.extend(("calibrated_predicted", None, scale) for scale in SCALES)
    grid.extend(("max_calibrated_fusion", None, scale) for scale in SCALES)
    return grid


def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            float(row["collision_rate"]),
            -float(row["success_rate"]),
            float(row["avg_cumulative_risk"]),
            float(row["avg_path_length"]),
        ),
    )


def write_rows(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FUSION_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def print_top(rows: list[dict[str, Any]], limit: int = 10) -> None:
    print("\nStage-5.8 fusion sweep top 10:")
    for idx, row in enumerate(rows[:limit], start=1):
        print(
            f"{idx}. mode={row['mode']} alpha={row['alpha']} scale={row['scale']} "
            f"success={float(row['success_rate']):.6f} collision={float(row['collision_rate']):.6f} "
            f"timeout={float(row['timeout_rate']):.6f} risk={float(row['avg_cumulative_risk']):.6f} "
            f"path={float(row['avg_path_length']):.6f} steps={float(row['avg_steps']):.6f}"
        )


def run_sweep(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.episodes <= 0:
        raise ValueError(f"--episodes must be positive, got {args.episodes}")
    base_config = load_config(args.config)
    hard = load_hard_config(args.hard_config)
    config = apply_hard_eval_config(base_config, hard)
    base_seed = int(config.get("data", {}).get("seed", config.get("seed", 42)))
    seeds = [base_seed + idx for idx in range(args.episodes)]
    device = resolve_device(args.device)
    model = build_loaded_model(config, args.checkpoint, device)

    rows = []
    for mode, alpha, scale in fusion_grid():
        summary, _ = evaluate_fusion_setting(
            mode=mode,
            alpha=alpha,
            scale=scale,
            config=config,
            hard=hard,
            seeds=seeds,
            device=device,
            model=model,
            verbose=False,
        )
        rows.append(summary)
    return sort_rows(rows)


def main() -> None:
    args = parse_args()
    rows = run_sweep(args)
    write_rows(rows, args.output)
    print_top(rows)
    print(f"Saved Stage-5.8 fusion sweep to: {args.output}")


if __name__ == "__main__":
    main()
