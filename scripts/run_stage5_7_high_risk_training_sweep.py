from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = PROJECT_ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from prism.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PRISM Stage-5.7 high-risk training sweep.")
    parser.add_argument("--configs", nargs="+", type=Path, required=True, help="Sweep training configs.")
    parser.add_argument("--hard-config", type=Path, default=Path("configs/toy_eval_hard.yaml"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--calibration-episodes", type=int, default=20)
    parser.add_argument("--baseline-episodes", type=int, default=100)
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
    pythonpath = str(PROJECT_PARENT)
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, env=env)


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def config_name(config_path: Path) -> str:
    return config_path.stem


def checkpoint_path(config: dict[str, Any]) -> Path:
    return Path(config.get("checkpoint_dir", "outputs/checkpoints")) / "best.pt"


def read_best_val_loss(path: Path) -> float:
    full_path = repo_path(path)
    if not full_path.exists():
        return float("nan")
    checkpoint = torch.load(full_path, map_location="cpu")
    return float(checkpoint.get("best_val_loss", float("nan")))


def read_prediction_metrics(path: Path) -> tuple[float, float]:
    rows = read_csv(path)
    if not rows:
        return float("nan"), float("nan")
    maes = [as_float(row["mae"]) for row in rows]
    rmses = [as_float(row["rmse"]) for row in rows]
    return mean(maes), math.sqrt(mean([value * value for value in rmses]))


def read_csv(path: Path) -> list[dict[str, str]]:
    full_path = repo_path(path)
    with full_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def mean(values: list[float]) -> float:
    values = [value for value in values if not math.isnan(value)]
    if not values:
        return float("nan")
    return sum(values) / len(values)


def parse_calibration_summary(path: Path) -> dict[str, float]:
    full_path = repo_path(path)
    result: dict[str, float] = {}
    for line in full_path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key.startswith("overall_"):
            result[key] = as_float(value.strip())
    return result


def high_risk_sample_stats(config: dict[str, Any]) -> dict[str, float]:
    npz_path = repo_path(config["train_npz"])
    with np.load(npz_path) as data:
        target = data["target"]
        threshold = float(config.get("high_risk_sample_threshold", 0.5))
        high_mask = (target > threshold).reshape(target.shape[0], -1).any(axis=1)
    high_count = int(high_mask.sum())
    total = int(high_mask.size)
    return {
        "train_total_samples": float(total),
        "train_high_risk_sample_count": float(high_count),
        "train_high_risk_sample_ratio": high_count / max(1, total),
    }


def copy_stage54_outputs(name: str) -> dict[str, Path]:
    result_dir = PROJECT_ROOT / "outputs/results/stage5_7"
    viz_dir = PROJECT_ROOT / "outputs/visualizations/stage5_7"
    result_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    copies = {
        "calibration_csv": result_dir / f"{name}_prediction_calibration.csv",
        "compare_csv": result_dir / f"{name}_current_vs_predicted.csv",
        "summary_txt": result_dir / f"{name}_calibration_summary.txt",
        "seed0_png": viz_dir / f"{name}_prediction_calibration_seed0.png",
        "failure_png": viz_dir / f"{name}_prediction_calibration_failure.png",
    }
    sources = {
        "calibration_csv": PROJECT_ROOT / "outputs/results/stage5_4_prediction_calibration.csv",
        "compare_csv": PROJECT_ROOT / "outputs/results/stage5_4_current_vs_predicted.csv",
        "summary_txt": PROJECT_ROOT / "outputs/results/stage5_4_calibration_summary.txt",
        "seed0_png": PROJECT_ROOT / "outputs/visualizations/stage5_4_prediction_calibration_seed0.png",
        "failure_png": PROJECT_ROOT / "outputs/visualizations/stage5_4_prediction_calibration_failure.png",
    }
    for key, source in sources.items():
        if source.exists():
            shutil.copy2(source, copies[key])
    return copies


def write_summary(rows: list[dict[str, Any]], output: Path) -> None:
    output = repo_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "config_name",
        "weighting_mode",
        "high_risk_weight",
        "hard_extra_weight",
        "use_high_risk_oversampling",
        "epochs",
        "best_val_loss",
        "overall_mae",
        "overall_rmse",
        "mu_recall_05",
        "safe_recall_05",
        "safe_risk_p95",
        "true_risk_p95",
        "current_mae",
        "mu_mae",
        "safe_mae",
        "sigma_error_corr",
        "train_total_samples",
        "train_high_risk_sample_count",
        "train_high_risk_sample_ratio",
        "calibration_passed",
        "scale_improved_but_recall_failed",
        "checkpoint_path",
        "calibration_summary_path",
    ]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -as_float(row["safe_recall_05"]),
            -as_float(row["mu_recall_05"]),
            abs(as_float(row["safe_risk_p95"]) - as_float(row["true_risk_p95"])),
            as_float(row["safe_mae"]),
        ),
    )


def package_results(configs: list[Path], top_row: dict[str, Any] | None, baseline_ran: bool) -> None:
    items = [
        "outputs/results/stage5_7_high_risk_training_sweep.csv",
        "README.md",
        *[str(path) for path in configs],
    ]
    if top_row is not None:
        name = str(top_row["config_name"])
        items.extend(
            [
                f"outputs/results/stage5_7/{name}_prediction_calibration.csv",
                f"outputs/results/stage5_7/{name}_current_vs_predicted.csv",
                f"outputs/results/stage5_7/{name}_calibration_summary.txt",
                f"outputs/visualizations/stage5_7/{name}_prediction_calibration_seed0.png",
                f"outputs/visualizations/stage5_7/{name}_prediction_calibration_failure.png",
            ]
        )
    if baseline_ran:
        items.extend(
            [
                "outputs/results/stage5_hard_baseline_results.csv",
                "outputs/results/stage5_hard_baseline_episode_results.csv",
                "outputs/visualizations/stage5_hard_baseline_comparison.png",
            ]
        )
    run_step(
        [
            sys.executable,
            "scripts/package_results.py",
            "--output",
            "outputs/prism_stage5_7_high_risk_sweep.zip",
            "--items",
            *items,
            "--exclude-protected",
        ]
    )


def run_one_config(config_path: Path, hard_config: Path, epochs: int, calibration_episodes: int, device: str) -> dict[str, Any]:
    name = config_name(config_path)
    config = load_config(config_path)
    stats = high_risk_sample_stats(config)
    ckpt = checkpoint_path(config)
    prediction_csv = Path("outputs/results/stage5_7") / f"{name}_prediction_metrics.csv"

    run_step([sys.executable, "scripts/train.py", "--config", str(config_path), "--epochs", str(epochs)])
    run_step(
        [
            sys.executable,
            "scripts/evaluate_prediction_on_toy.py",
            "--config",
            str(config_path),
            "--checkpoint",
            str(ckpt),
            "--output",
            str(prediction_csv),
        ]
    )
    run_step(
        [
            sys.executable,
            "scripts/run_stage5_4_calibration.py",
            "--config",
            str(config_path),
            "--hard-config",
            str(hard_config),
            "--checkpoint",
            str(ckpt),
            "--episodes",
            str(calibration_episodes),
            "--device",
            device,
        ]
    )
    copies = copy_stage54_outputs(name)
    calibration = parse_calibration_summary(copies["summary_txt"])
    overall_mae, overall_rmse = read_prediction_metrics(prediction_csv)
    mu_recall = calibration.get("overall_mu_recall_05", float("nan"))
    safe_recall = calibration.get("overall_safe_risk_recall_05", float("nan"))
    safe_p95 = calibration.get("overall_safe_risk_p95", float("nan"))
    true_p95 = calibration.get("overall_true_risk_p95", float("nan"))
    calibration_passed = bool(mu_recall > 0.0 or safe_recall > 0.0)
    scale_improved = bool(not calibration_passed and safe_p95 >= 0.22)

    return {
        "config_name": name,
        "weighting_mode": config.get("weighting_mode", ""),
        "high_risk_weight": config.get("high_risk_weight", ""),
        "hard_extra_weight": config.get("hard_extra_weight", ""),
        "use_high_risk_oversampling": bool(config.get("use_high_risk_oversampling", False)),
        "epochs": epochs,
        "best_val_loss": read_best_val_loss(ckpt),
        "overall_mae": overall_mae,
        "overall_rmse": overall_rmse,
        "mu_recall_05": mu_recall,
        "safe_recall_05": safe_recall,
        "safe_risk_p95": safe_p95,
        "true_risk_p95": true_p95,
        "current_mae": calibration.get("overall_current_mae", float("nan")),
        "mu_mae": calibration.get("overall_mu_mae", float("nan")),
        "safe_mae": calibration.get("overall_safe_risk_mae", float("nan")),
        "sigma_error_corr": calibration.get("overall_sigma_error_corr", float("nan")),
        "checkpoint_path": str(ckpt),
        "calibration_summary_path": str(copies["summary_txt"].relative_to(PROJECT_ROOT)),
        "calibration_passed": calibration_passed,
        "scale_improved_but_recall_failed": scale_improved,
        **stats,
    }


def print_top(rows: list[dict[str, Any]]) -> None:
    print("\nStage-5.7 top 3 configs:")
    for rank, row in enumerate(rows[:3], start=1):
        print(
            f"{rank}. {row['config_name']} mode={row['weighting_mode']} "
            f"weight={row['high_risk_weight']} os={row['use_high_risk_oversampling']} "
            f"mu_recall_05={as_float(row['mu_recall_05']):.6f} "
            f"safe_recall_05={as_float(row['safe_recall_05']):.6f} "
            f"safe_p95={as_float(row['safe_risk_p95']):.6f} "
            f"true_p95={as_float(row['true_risk_p95']):.6f} "
            f"safe_mae={as_float(row['safe_mae']):.6f}"
        )


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError(f"--epochs must be positive, got {args.epochs}")
    if args.calibration_episodes <= 0:
        raise ValueError(f"--calibration-episodes must be positive, got {args.calibration_episodes}")

    rows = []
    for index, config_path in enumerate(args.configs, start=1):
        print(f"\nStage-5.7 config {index}/{len(args.configs)}: {config_path}", flush=True)
        rows.append(
            run_one_config(
                config_path=config_path,
                hard_config=args.hard_config,
                epochs=args.epochs,
                calibration_episodes=args.calibration_episodes,
                device=args.device,
            )
        )

    sorted_rows = sort_rows(rows)
    summary_csv = Path("outputs/results/stage5_7_high_risk_training_sweep.csv")
    write_summary(sorted_rows, summary_csv)
    print_top(sorted_rows)

    top_row = sorted_rows[0] if sorted_rows else None
    baseline_ran = False
    if top_row and bool(top_row["calibration_passed"]):
        run_step(
            [
                sys.executable,
                "scripts/run_stage5_1_hard_pipeline.py",
                "--config",
                str(args.configs[[config_name(path) for path in args.configs].index(str(top_row["config_name"]))]),
                "--hard-config",
                str(args.hard_config),
                "--checkpoint",
                str(top_row["checkpoint_path"]),
                "--episodes",
                str(args.baseline_episodes),
                "--device",
                args.device,
            ]
        )
        baseline_ran = True
    else:
        print("No config reached recall_05 > 0; hard baseline is skipped.", flush=True)

    package_results(args.configs, top_row, baseline_ran)
    print(f"Saved Stage-5.7 sweep summary to: {summary_csv}")
    print("Saved Stage-5.7 package to: outputs/prism_stage5_7_high_risk_sweep.zip")


if __name__ == "__main__":
    main()
