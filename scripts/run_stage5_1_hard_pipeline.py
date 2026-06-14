from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS = ["goal_greedy", "current_risk", "mean_risk", "prism_no_propagation", "prism_full"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PRISM Stage-5.1 hard baseline and ablation pipeline.")
    parser.add_argument("--config", type=Path, default=Path("configs/toy_train.yaml"), help="Base model config.")
    parser.add_argument(
        "--hard-config",
        type=Path,
        default=Path("configs/toy_eval_hard.yaml"),
        help="Hard evaluation config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/checkpoints_toy/best.pt"),
        help="Path to trained PRISM checkpoint.",
    )
    parser.add_argument("--episodes", type=int, default=100, help="Number of hard episodes per method.")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Device to use for model inference.",
    )
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
    summary_csv = Path("outputs/results/stage5_hard_baseline_results.csv")
    episode_csv = Path("outputs/results/stage5_hard_baseline_episode_results.csv")
    plot_png = Path("outputs/visualizations/stage5_hard_baseline_comparison.png")
    zip_path = Path("outputs/prism_stage5_1_hard_results.zip")

    print("Stage-5.1 hard method list:", " ".join(METHODS))
    run_step(
        [
            sys.executable,
            "scripts/evaluate_baselines_hard.py",
            "--config",
            str(args.config),
            "--hard-config",
            str(args.hard_config),
            "--checkpoint",
            str(args.checkpoint),
            "--episodes",
            str(args.episodes),
            "--device",
            args.device,
            "--methods",
            *METHODS,
            "--summary-output",
            str(summary_csv),
            "--episode-output",
            str(episode_csv),
        ]
    )
    run_step(
        [
            sys.executable,
            "scripts/plot_baseline_results.py",
            "--csv",
            str(summary_csv),
            "--output",
            str(plot_png),
        ]
    )
    run_step(
        [
            sys.executable,
            "scripts/package_results.py",
            "--output",
            str(zip_path),
            "--items",
            str(summary_csv),
            str(episode_csv),
            str(plot_png),
            str(args.config),
            str(args.hard_config),
            "README.md",
            "--exclude-protected",
        ]
    )

    print("Stage-5.1 hard outputs:")
    print(f"stage5_hard_baseline_results.csv: {summary_csv}")
    print(f"stage5_hard_baseline_episode_results.csv: {episode_csv}")
    print(f"stage5_hard_baseline_comparison.png: {plot_png}")
    print(f"prism_stage5_1_hard_results.zip: {zip_path}")


if __name__ == "__main__":
    main()
