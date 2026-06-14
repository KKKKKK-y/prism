from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PRISM Stage-5.8 fusion pipeline.")
    parser.add_argument("--hard-config", type=Path, default=Path("configs/toy_eval_hard.yaml"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
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
    run_step(
        [
            sys.executable,
            "scripts/run_stage5_8_fusion_pipeline.py",
            "--hard-config",
            str(args.hard_config),
            "--episodes",
            str(args.episodes),
            "--device",
            args.device,
        ]
    )
    run_step(
        [
            sys.executable,
            "scripts/plot_fusion_results.py",
            "--input",
            "outputs/results/stage5_8_fusion_all_checkpoints.csv",
            "--output",
            "outputs/visualizations/stage5_8_fusion_comparison.png",
        ]
    )
    run_step(
        [
            sys.executable,
            "scripts/package_results.py",
            "--output",
            "outputs/prism_stage5_8_fusion_results.zip",
            "--items",
            "outputs/results/stage5_8_fusion_all_checkpoints.csv",
            "outputs/results/stage5_8_fusion_sweep_unweighted_hard_b.csv",
            "outputs/results/stage5_8_fusion_sweep_weighted_w20.csv",
            "outputs/results/stage5_8_fusion_sweep_weighted_w50.csv",
            "outputs/visualizations/stage5_8_fusion_comparison.png",
            "configs/toy_train_hard_level_b.yaml",
            "configs/toy_train_hard_weighted_w20.yaml",
            "configs/toy_train_hard_weighted_w50.yaml",
            str(args.hard_config),
            "README.md",
            "--exclude-protected",
        ]
    )
    print("Stage-5.8 outputs:")
    print("fusion_all_checkpoints: outputs/results/stage5_8_fusion_all_checkpoints.csv")
    print("fusion_plot: outputs/visualizations/stage5_8_fusion_comparison.png")
    print("package: outputs/prism_stage5_8_fusion_results.zip")


if __name__ == "__main__":
    main()
