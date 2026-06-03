from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PRISM Stage-4.3 toy training/evaluation pipeline.")
    parser.add_argument("--config", type=Path, default=Path("configs/toy_train.yaml"), help="Toy training YAML config.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/checkpoints_toy/best.pt"),
        help="Checkpoint path produced by the training step.",
    )
    parser.add_argument(
        "--training-csv",
        type=Path,
        default=Path("outputs/results/stage4_toy_training_log.csv"),
        help="Training log CSV path.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("outputs/results/stage4_toy_prediction_metrics.csv"),
        help="Prediction metrics CSV path.",
    )
    parser.add_argument(
        "--visualization-dir",
        type=Path,
        default=Path("outputs/visualizations"),
        help="Directory for Stage-4.3 figures.",
    )
    parser.add_argument("--debug", action="store_true", help="Run the train step with scripts/train.py --debug.")
    return parser.parse_args()


def run_step(command: list[str], project_root: Path) -> None:
    print(f"\nRunning: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=project_root, check=True)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    python = sys.executable

    train_cmd = [python, "scripts/train.py", "--config", str(args.config)]
    if args.debug:
        train_cmd.append("--debug")

    training_curve_path = args.visualization_dir / "stage4_toy_training_curve.png"
    prediction_metrics_path = args.visualization_dir / "stage4_toy_prediction_metrics.png"
    horizon_visualization_path = args.visualization_dir / "stage4_toy_prediction_horizons.png"

    steps = [
        train_cmd,
        [
            python,
            "scripts/evaluate_prediction_on_toy.py",
            "--config",
            str(args.config),
            "--checkpoint",
            str(args.checkpoint),
            "--output",
            str(args.metrics_csv),
        ],
        [
            python,
            "scripts/plot_training_curve.py",
            "--csv",
            str(args.training_csv),
            "--output",
            str(training_curve_path),
        ],
        [
            python,
            "scripts/plot_prediction_metrics.py",
            "--csv",
            str(args.metrics_csv),
            "--output",
            str(prediction_metrics_path),
        ],
        [
            python,
            "scripts/visualize_prediction.py",
            "--config",
            str(args.config),
            "--checkpoint",
            str(args.checkpoint),
            "--output",
            str(horizon_visualization_path),
            "--all_horizons",
        ],
    ]

    for step in steps:
        run_step(step, project_root)

    print("\nPRISM Stage-4.3 pipeline completed.", flush=True)


if __name__ == "__main__":
    main()
