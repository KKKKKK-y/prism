from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PRISM Stage-5.5 target generation alignment checks.")
    parser.add_argument("--config", type=Path, default=Path("configs/toy_train_hard_level_b.yaml"))
    parser.add_argument("--hard-config", type=Path, default=Path("configs/toy_eval_hard.yaml"))
    parser.add_argument("--npz", type=Path, default=Path("outputs/datasets_hard/toy_fire_train_hard.npz"))
    parser.add_argument("--episodes", type=int, default=5)
    return parser.parse_args()


def run_step(command: list[str]) -> None:
    print(f"\nRunning: {' '.join(command)}", flush=True)
    env = dict(os.environ)
    pythonpath = str(PROJECT_ROOT.parent)
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, env=env)


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError(f"--episodes must be positive, got {args.episodes}")

    distribution_txt = Path("outputs/results/stage5_5_hard_target_distribution.txt")
    alignment_csv = Path("outputs/results/stage5_5_target_rollout_alignment.csv")
    alignment_summary = Path("outputs/results/stage5_5_target_rollout_alignment_summary.txt")
    alignment_png = Path("outputs/visualizations/stage5_5_target_alignment_seed0.png")
    shift_png = Path("outputs/visualizations/stage5_5_target_horizon_shift_seed0.png")
    zip_path = Path("outputs/prism_stage5_5_target_alignment.zip")

    run_step(
        [
            sys.executable,
            "scripts/inspect_dataset_targets.py",
            "--npz",
            str(args.npz),
            "--output",
            str(distribution_txt),
        ]
    )
    run_step(
        [
            sys.executable,
            "scripts/test_target_rollout_alignment.py",
            "--config",
            str(args.config),
            "--hard-config",
            str(args.hard_config),
            "--episodes",
            str(args.episodes),
            "--output",
            str(alignment_csv),
            "--summary-output",
            str(alignment_summary),
        ]
    )
    run_step(
        [
            sys.executable,
            "scripts/visualize_target_alignment.py",
            "--config",
            str(args.config),
            "--hard-config",
            str(args.hard_config),
            "--seed",
            "0",
            "--output",
            str(alignment_png),
            "--shift-output",
            str(shift_png),
        ]
    )
    run_step(
        [
            sys.executable,
            "scripts/package_results.py",
            "--output",
            str(zip_path),
            "--items",
            str(distribution_txt),
            str(alignment_csv),
            str(alignment_summary),
            str(alignment_png),
            str(shift_png),
            str(args.config),
            str(args.hard_config),
            "README.md",
            "--exclude-protected",
        ]
    )

    print("Stage-5.5 outputs:")
    print(f"stage5_5_hard_target_distribution.txt: {distribution_txt}")
    print(f"stage5_5_target_rollout_alignment.csv: {alignment_csv}")
    print(f"stage5_5_target_rollout_alignment_summary.txt: {alignment_summary}")
    print(f"stage5_5_target_alignment_seed0.png: {alignment_png}")
    print(f"stage5_5_target_horizon_shift_seed0.png: {shift_png}")
    print(f"prism_stage5_5_target_alignment.zip: {zip_path}")


if __name__ == "__main__":
    main()
