from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = PROJECT_ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small reviewer PRISM demo from scratch.")
    parser.add_argument("--config", type=Path, default=Path("configs/reviewer_demo.yaml"))
    parser.add_argument("--epochs", type=int, default=None, help="Override demo training epochs.")
    parser.add_argument("--closed-loop-episodes", type=int, default=None)
    parser.add_argument("--fusion-episodes", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.expanduser().open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def project_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    pieces = [str(PROJECT_PARENT)]
    if existing:
        pieces.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pieces)
    return env


def run_command(args: list[str]) -> None:
    command = [sys.executable, *args]
    print(f"\n$ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=project_env(), check=True)


def print_environment() -> None:
    print("Reviewer demo environment:")
    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_version: {torch.version.cuda}")
        print(f"cuda_device: {torch.cuda.get_device_name(0)}")
    else:
        print("cuda_device: none")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_npz(path: Path) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    with np.load(path) as data:
        obs_shape = tuple(int(v) for v in data["obs"].shape)
        target_shape = tuple(int(v) for v in data["target"].shape)
    return obs_shape[0], obs_shape, target_shape


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def training_metrics(path: Path) -> tuple[float | None, float | None]:
    rows = read_csv(path)
    values = []
    for row in rows:
        value = row.get("val_loss", "")
        if value != "":
            values.append(float(value))
    if not values:
        return None, None
    return min(values), values[-1]


def prediction_metrics(path: Path) -> tuple[float | None, float | None]:
    rows = read_csv(path)
    if not rows:
        return None, None
    mae = sum(float(row["mae"]) for row in rows) / len(rows)
    rmse = sum(float(row["rmse"]) for row in rows) / len(rows)
    return mae, rmse


def closed_loop_metrics(path: Path) -> tuple[float | None, float | None]:
    rows = read_csv(path)
    if not rows:
        return None, None
    n = len(rows)
    success = sum(int(row["success"]) for row in rows) / n
    collision = sum(int(row["collision"]) for row in rows) / n
    return success, collision


def fusion_metrics(path: Path) -> tuple[float | None, float | None]:
    rows = read_csv(path)
    for row in rows:
        if row.get("method") == "alpha_fusion_0.4":
            return float(row["success_rate"]), float(row["collision_rate"])
    return None, None


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_temp_final_config(config_path: Path, checkpoint_path: Path, config: dict[str, Any], episodes: int) -> Path:
    planner = config.get("planner", {}) if isinstance(config.get("planner"), dict) else {}
    temp_path = PROJECT_ROOT / "outputs" / "results" / "reviewer_demo_final_fusion.yaml"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "planner": {
            "final_method": "alpha_fusion",
            "alpha": float(planner.get("alpha", 0.4)),
            "lambda_u": float(config.get("lambda_u", 0.5)),
            "uncertainty_alpha": float(config.get("uncertainty_alpha", 0.7)),
            "num_mc_samples": int(config.get("num_mc_samples", 3)),
        },
        "model": {
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
        },
        "eval": {
            "hard_config": "configs/toy_eval_hard.yaml",
            "episodes": int(episodes),
        },
    }
    with temp_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return temp_path


def main() -> None:
    args = parse_args()
    config_path = args.config
    config = load_yaml(config_path)
    seed = int(config.get("seed", config.get("data", {}).get("seed", 42)))
    set_seed(seed)
    print_environment()
    print(f"seed: {seed}")

    train_npz = PROJECT_ROOT / Path(config.get("train_npz", "outputs/datasets_reviewer_demo/toy_fire_train.npz"))
    val_npz = PROJECT_ROOT / Path(config.get("val_npz", "outputs/datasets_reviewer_demo/toy_fire_val.npz"))
    test_npz = PROJECT_ROOT / Path(config.get("test_npz", "outputs/datasets_reviewer_demo/toy_fire_test.npz"))
    dataset_dir = train_npz.parent
    checkpoint_path = PROJECT_ROOT / Path(config.get("checkpoint_dir", "outputs/checkpoints_reviewer_demo")) / "best.pt"
    results_dir = PROJECT_ROOT / Path(config.get("results_dir", "outputs/results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    epochs = int(args.epochs if args.epochs is not None else config.get("epochs", 3))
    closed_loop_episodes = int(
        args.closed_loop_episodes if args.closed_loop_episodes is not None else config.get("closed_loop_episodes", 5)
    )
    fusion_episodes = int(args.fusion_episodes if args.fusion_episodes is not None else config.get("fusion_episodes", 5))

    print("\nphase=dataset_generation", flush=True)
    run_command(
        [
            "scripts/generate_toy_dataset.py",
            "--config",
            str(config_path),
            "--train_episodes",
            str(int(config.get("train_episodes", 10))),
            "--val_episodes",
            str(int(config.get("val_episodes", 2))),
            "--test_episodes",
            str(int(config.get("test_episodes", 2))),
            "--output_dir",
            str(dataset_dir),
        ]
    )

    train_count, train_obs_shape, train_target_shape = count_npz(train_npz)
    val_count, val_obs_shape, val_target_shape = count_npz(val_npz)
    test_count, test_obs_shape, test_target_shape = count_npz(test_npz)
    print(f"train_shape obs={train_obs_shape} target={train_target_shape}")
    print(f"val_shape obs={val_obs_shape} target={val_target_shape}")
    print(f"test_shape obs={test_obs_shape} target={test_target_shape}")

    print("\nphase=training", flush=True)
    run_command(["scripts/train.py", "--config", str(config_path), "--epochs", str(epochs)])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Expected checkpoint was not created: {checkpoint_path}")

    prediction_csv = results_dir / "reviewer_demo_prediction_metrics.csv"
    closed_loop_csv = results_dir / "reviewer_demo_closed_loop_results.csv"
    fusion_csv = results_dir / "reviewer_demo_fusion_results.csv"
    fusion_episode_csv = results_dir / "reviewer_demo_fusion_episode_results.csv"

    print("\nphase=prediction_eval", flush=True)
    run_command(
        [
            "scripts/evaluate_prediction_on_toy.py",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(prediction_csv),
        ]
    )

    print("\nphase=closed_loop_eval", flush=True)
    run_command(
        [
            "scripts/evaluate_closed_loop.py",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--episodes",
            str(closed_loop_episodes),
            "--output",
            str(closed_loop_csv),
            "--device",
            args.device,
        ]
    )

    print("\nphase=fusion_planner_demo", flush=True)
    final_config_path = write_temp_final_config(config_path, checkpoint_path, config, fusion_episodes)
    run_command(
        [
            "scripts/run_stage5_9_final_fusion_eval.py",
            "--config",
            str(final_config_path),
            "--episodes",
            str(fusion_episodes),
            "--summary-output",
            str(fusion_csv),
            "--episode-output",
            str(fusion_episode_csv),
            "--device",
            args.device,
        ]
    )

    best_val_loss, final_val_loss = training_metrics(PROJECT_ROOT / Path(config.get("training_log_csv", "")))
    pred_mae, pred_rmse = prediction_metrics(prediction_csv)
    closed_success, closed_collision = closed_loop_metrics(closed_loop_csv)
    fusion_success, fusion_collision = fusion_metrics(fusion_csv)

    summary_path = results_dir / "reviewer_demo_summary.txt"
    lines = [
        "PRISM reviewer demo summary",
        "===========================",
        f"config: {config_path}",
        f"seed: {seed}",
        f"device_requested: {args.device}",
        f"train_samples: {train_count}",
        f"val_samples: {val_count}",
        f"test_samples: {test_count}",
        f"epochs: {epochs}",
        f"best_val_loss: {fmt(best_val_loss)}",
        f"final_val_loss: {fmt(final_val_loss)}",
        f"prediction_mae: {fmt(pred_mae)}",
        f"prediction_rmse: {fmt(pred_rmse)}",
        f"closed_loop_success_rate: {fmt(closed_success)}",
        f"closed_loop_collision_rate: {fmt(closed_collision)}",
        f"fusion_success_rate: {fmt(fusion_success)}",
        f"fusion_collision_rate: {fmt(fusion_collision)}",
        "",
        "Output paths:",
        f"checkpoint: {checkpoint_path}",
        f"prediction_metrics: {prediction_csv}",
        f"closed_loop_results: {closed_loop_csv}",
        f"fusion_results: {fusion_csv}",
        f"summary: {summary_path}",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved reviewer demo summary to: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nReviewer demo failed with traceback:", file=sys.stderr)
        traceback.print_exc()
        raise
