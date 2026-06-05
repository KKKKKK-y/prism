from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = PROJECT_ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from prism.config import load_config
from prism.trainers.trainer import select_device


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
    parser.add_argument(
        "--closed-loop-output",
        type=Path,
        default=Path("outputs/results/stage4_closed_loop_results.csv"),
        help="Closed-loop evaluation CSV path.",
    )
    parser.add_argument(
        "--closed-loop-episodes",
        type=int,
        default=100,
        help="Number of closed-loop evaluation episodes.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/results/formal_run_summary.txt"),
        help="Formal run summary output path.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--debug", action="store_true", help="Run the train step with scripts/train.py --debug.")
    return parser.parse_args()


def run_step(command: list[str], project_root: Path) -> None:
    print(f"\nRunning: {' '.join(command)}", flush=True)
    env = dict(os.environ)
    pythonpath = str(project_root.parent)
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    subprocess.run(command, cwd=project_root, check=True, env=env)


def rel(path: Path, project_root: Path) -> Path:
    try:
        return path.relative_to(project_root)
    except ValueError:
        return path


def resolve_path(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def command_output(command: list[str], project_root: Path) -> str:
    try:
        result = subprocess.run(command, cwd=project_root, check=True, capture_output=True, text=True)
    except Exception:
        return "N/A"
    return result.stdout.strip() or "N/A"


def dataset_size(path: Path) -> str:
    if not path.exists():
        return "N/A"
    try:
        with np.load(path) as data:
            return str(int(data["obs"].shape[0]))
    except Exception:
        return "N/A"


def read_last_training_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None
    return rows[-1] if rows else None


def best_val_loss(checkpoint_path: Path, training_csv: Path) -> str:
    if checkpoint_path.exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(checkpoint, dict) and "best_val_loss" in checkpoint:
                return f"{float(checkpoint['best_val_loss']):.6f}"
        except Exception:
            pass
    row = read_last_training_row(training_csv)
    if row and row.get("val_loss"):
        return row["val_loss"]
    return "N/A"


def prediction_metrics(path: Path) -> tuple[str, str, list[dict[str, str]]]:
    if not path.exists():
        return "N/A", "N/A", []
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return "N/A", "N/A", []
        mae_values = [float(row["mae"]) for row in rows if row.get("mae")]
        rmse_values = [float(row["rmse"]) for row in rows if row.get("rmse")]
        overall_mae = f"{sum(mae_values) / len(mae_values):.6f}" if mae_values else "N/A"
        overall_rmse = f"{(sum(value * value for value in rmse_values) / len(rmse_values)) ** 0.5:.6f}" if rmse_values else "N/A"
        return overall_mae, overall_rmse, rows
    except Exception:
        return "N/A", "N/A", []


def closed_loop_summary(path: Path) -> dict[str, str]:
    keys = ["success_rate", "collision_rate", "avg_cumulative_risk", "avg_path_length", "avg_steps"]
    if not path.exists():
        return {key: "N/A" for key in keys}
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return {key: "N/A" for key in keys}
        n = len(rows)
        return {
            "success_rate": f"{sum(int(row['success']) for row in rows) / n:.6f}",
            "collision_rate": f"{sum(int(row['collision']) for row in rows) / n:.6f}",
            "avg_cumulative_risk": f"{sum(float(row['cumulative_risk']) for row in rows) / n:.6f}",
            "avg_path_length": f"{sum(float(row['path_length']) for row in rows) / n:.6f}",
            "avg_steps": f"{sum(float(row['steps']) for row in rows) / n:.6f}",
        }
    except Exception:
        return {key: "N/A" for key in keys}


def print_formal_config_summary(config: dict[str, Any], args: argparse.Namespace, project_root: Path) -> None:
    device = select_device()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    checkpoint_dir = Path(config.get("checkpoint_dir", "outputs/checkpoints_toy"))
    output_dir = Path(config.get("output_dir", "outputs"))
    rows = {
        "train_episodes": "external dataset step or existing NPZ",
        "val_episodes": "external dataset step or existing NPZ",
        "test_episodes": "external dataset step or existing NPZ",
        "epochs": str(config.get("epochs", "N/A")),
        "batch_size": str(config.get("batch_size", "N/A")),
        "device": str(device),
        "GPU name": gpu_name,
        "dataset output path": str(rel(resolve_path(Path(config.get("train_npz", "outputs/datasets/toy_fire_train.npz")), project_root).parent, project_root)),
        "checkpoint path": str(checkpoint_dir / "best.pt"),
        "results path": str(output_dir / "results"),
        "visualization path": str(args.visualization_dir),
    }
    print("\nFormal config summary:", flush=True)
    for key, value in rows.items():
        print(f"- {key}: {value}", flush=True)
    if device.type != "cuda":
        print("WARNING: Formal pipeline is running without CUDA. This may be slow.", flush=True)


def ensure_debug_dataset(config: dict[str, Any], project_root: Path, python: str) -> None:
    required = [
        resolve_path(Path(config.get("train_npz", "outputs/datasets/toy_fire_train.npz")), project_root),
        resolve_path(Path(config.get("val_npz", "outputs/datasets/toy_fire_val.npz")), project_root),
        resolve_path(Path(config.get("test_npz", "outputs/datasets/toy_fire_test.npz")), project_root),
    ]
    if all(path.exists() for path in required):
        return
    print("Debug dataset files are missing. Generating a tiny debug dataset.", flush=True)
    run_step(
        [
            python,
            "scripts/generate_toy_dataset.py",
            "--config",
            "configs/toy_train.yaml",
            "--train_episodes",
            "2",
            "--val_episodes",
            "1",
            "--test_episodes",
            "1",
        ],
        project_root,
    )


def write_summary(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    project_root: Path,
    start_time: float,
) -> None:
    summary_path = resolve_path(args.summary_output, project_root)
    metrics_csv = resolve_path(args.metrics_csv, project_root)
    training_csv = resolve_path(args.training_csv, project_root)
    checkpoint_path = resolve_path(args.checkpoint, project_root)
    closed_loop_csv = resolve_path(args.closed_loop_output, project_root)

    overall_mae, overall_rmse, horizon_rows = prediction_metrics(metrics_csv)
    closed_loop = closed_loop_summary(closed_loop_csv)
    config_json = json.dumps(config, indent=2, sort_keys=True)
    elapsed_seconds = time.time() - start_time
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"

    lines = [
        "PRISM Formal Run Summary",
        "========================",
        f"run_time_seconds: {elapsed_seconds:.2f}",
        f"git_commit_hash: {command_output(['git', 'rev-parse', 'HEAD'], project_root)}",
        f"python_version: {platform.python_version()}",
        f"pytorch_version: {torch.__version__}",
        f"cuda_available: {cuda_available}",
        f"gpu_name: {gpu_name}",
        "",
        "Training Config",
        "---------------",
        config_json,
        "",
        "Dataset Size",
        "------------",
        f"train_samples: {dataset_size(resolve_path(Path(config.get('train_npz', 'outputs/datasets/toy_fire_train.npz')), project_root))}",
        f"val_samples: {dataset_size(resolve_path(Path(config.get('val_npz', 'outputs/datasets/toy_fire_val.npz')), project_root))}",
        f"test_samples: {dataset_size(resolve_path(Path(config.get('test_npz', 'outputs/datasets/toy_fire_test.npz')), project_root))}",
        "",
        "Final Training Result",
        "---------------------",
        f"best_val_loss: {best_val_loss(checkpoint_path, training_csv)}",
        "",
        "Prediction Metrics",
        "------------------",
        f"overall_MAE: {overall_mae}",
        f"overall_RMSE: {overall_rmse}",
    ]
    if horizon_rows:
        for row in horizon_rows:
            lines.append(f"horizon_t+{row.get('horizon', 'N/A')}_MAE: {row.get('mae', 'N/A')}")
            lines.append(f"horizon_t+{row.get('horizon', 'N/A')}_RMSE: {row.get('rmse', 'N/A')}")
    else:
        lines.append("horizon-wise_MAE/RMSE: N/A")

    lines.extend(
        [
            "",
            "Closed-Loop Evaluation",
            "----------------------",
            f"success_rate: {closed_loop['success_rate']}",
            f"collision_rate: {closed_loop['collision_rate']}",
            f"avg_cumulative_risk: {closed_loop['avg_cumulative_risk']}",
            f"avg_path_length: {closed_loop['avg_path_length']}",
            f"avg_steps: {closed_loop['avg_steps']}",
            "",
        ]
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved formal run summary to: {rel(summary_path, project_root)}", flush=True)


def main() -> None:
    start_time = time.time()
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    python = sys.executable
    config = load_config(args.config)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    print_formal_config_summary(config, args, project_root)

    train_cmd = [python, "scripts/train.py", "--config", str(args.config)]
    if args.epochs is not None:
        train_cmd.extend(["--epochs", str(args.epochs)])
    if args.debug:
        ensure_debug_dataset(config, project_root, python)
        train_cmd.append("--debug")
        args.closed_loop_episodes = min(args.closed_loop_episodes, 2)

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
        [
            python,
            "scripts/evaluate_closed_loop.py",
            "--config",
            str(args.config),
            "--checkpoint",
            str(args.checkpoint),
            "--episodes",
            str(args.closed_loop_episodes),
            "--output",
            str(args.closed_loop_output),
        ],
    ]

    for step in steps:
        run_step(step, project_root)

    write_summary(args=args, config=config, project_root=project_root, start_time=start_time)
    print("\nPRISM Stage-4.3 pipeline completed.", flush=True)


if __name__ == "__main__":
    main()
