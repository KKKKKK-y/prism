from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = PROJECT_ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from prism.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PRISM Stage-5.6 high-risk weighted hard training pipeline.")
    parser.add_argument("--config", type=Path, default=Path("configs/toy_train_hard_weighted.yaml"))
    parser.add_argument("--hard-config", type=Path, default=Path("configs/toy_eval_hard.yaml"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--calibration-episodes", type=int, default=50)
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


def resolve_checkpoint(config: dict) -> Path:
    checkpoint_dir = Path(config.get("checkpoint_dir", "outputs/checkpoints_toy_hard_weighted"))
    return checkpoint_dir / "best.pt"


def check_hard_datasets(config: dict) -> None:
    required_keys = ("train_npz", "val_npz", "test_npz")
    missing: list[Path] = []
    for key in required_keys:
        path = Path(config[key])
        path = path if path.is_absolute() else PROJECT_ROOT / path
        if not path.exists():
            missing.append(path)
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Stage-5.6 requires existing hard datasets. Missing:\n{missing_text}")


def read_recall(summary_path: Path) -> tuple[float, float]:
    mu_recall = float("nan")
    safe_recall = float("nan")
    if not summary_path.exists():
        raise FileNotFoundError(f"Calibration summary not found: {summary_path}")
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("overall_mu_recall_05:"):
            mu_recall = float(line.split(":", 1)[1].strip())
        elif line.startswith("overall_safe_risk_recall_05:"):
            safe_recall = float(line.split(":", 1)[1].strip())
    return mu_recall, safe_recall


def package_results(config_path: Path, hard_config_path: Path, include_baseline: bool) -> None:
    items = [
        "outputs/results/stage4_toy_hard_weighted_training_log.csv",
        "outputs/results/stage4_toy_hard_weighted_prediction_metrics.csv",
        "outputs/results/stage5_4_prediction_calibration.csv",
        "outputs/results/stage5_4_current_vs_predicted.csv",
        "outputs/results/stage5_4_calibration_summary.txt",
        "outputs/visualizations/stage5_4_prediction_calibration_seed0.png",
        "outputs/visualizations/stage5_4_prediction_calibration_failure.png",
        "outputs/results/stage4_toy_hard_weighted_closed_loop_results.csv",
        str(config_path),
        str(hard_config_path),
        "README.md",
    ]
    if include_baseline:
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
            "outputs/prism_stage5_6_weighted_results.zip",
            "--items",
            *items,
            "--exclude-protected",
        ]
    )


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError(f"--epochs must be positive, got {args.epochs}")
    if args.calibration_episodes <= 0:
        raise ValueError(f"--calibration-episodes must be positive, got {args.calibration_episodes}")
    if args.baseline_episodes <= 0:
        raise ValueError(f"--baseline-episodes must be positive, got {args.baseline_episodes}")

    config = load_config(args.config)
    check_hard_datasets(config)
    checkpoint = resolve_checkpoint(config)

    prediction_csv = Path("outputs/results/stage4_toy_hard_weighted_prediction_metrics.csv")
    closed_loop_csv = Path("outputs/results/stage4_toy_hard_weighted_closed_loop_results.csv")
    calibration_summary = PROJECT_ROOT / "outputs/results/stage5_4_calibration_summary.txt"

    print("Stage-5.6 weighted training config:")
    print(f"config: {args.config}")
    print(f"hard_config: {args.hard_config}")
    print(f"checkpoint: {checkpoint}")
    print(f"epochs: {args.epochs}")
    print("Dataset generation is intentionally skipped; existing hard npz files are reused.", flush=True)

    run_step([sys.executable, "scripts/train.py", "--config", str(args.config), "--epochs", str(args.epochs)])
    run_step(
        [
            sys.executable,
            "scripts/evaluate_prediction_on_toy.py",
            "--config",
            str(args.config),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(prediction_csv),
        ]
    )
    run_step(
        [
            sys.executable,
            "scripts/run_stage5_4_calibration.py",
            "--config",
            str(args.config),
            "--hard-config",
            str(args.hard_config),
            "--checkpoint",
            str(checkpoint),
            "--episodes",
            str(args.calibration_episodes),
            "--device",
            args.device,
        ]
    )

    mu_recall, safe_recall = read_recall(calibration_summary)
    print(f"Stage-5.6 calibration recall: mu_recall_05={mu_recall:.6f}, safe_recall_05={safe_recall:.6f}")
    if mu_recall <= 0.0 and safe_recall <= 0.0:
        print(
            "Stage-5.6 stopping before hard baseline because both high-risk recall metrics are still 0.",
            flush=True,
        )
        package_results(args.config, args.hard_config, include_baseline=False)
        return

    run_step(
        [
            sys.executable,
            "scripts/evaluate_closed_loop.py",
            "--config",
            str(args.config),
            "--checkpoint",
            str(checkpoint),
            "--episodes",
            str(args.baseline_episodes),
            "--output",
            str(closed_loop_csv),
            "--device",
            args.device,
        ]
    )
    run_step(
        [
            sys.executable,
            "scripts/run_stage5_1_hard_pipeline.py",
            "--config",
            str(args.config),
            "--hard-config",
            str(args.hard_config),
            "--checkpoint",
            str(checkpoint),
            "--episodes",
            str(args.baseline_episodes),
            "--device",
            args.device,
        ]
    )
    package_results(args.config, args.hard_config, include_baseline=True)

    print("Stage-5.6 outputs:")
    print("weighted_prediction_metrics: outputs/results/stage4_toy_hard_weighted_prediction_metrics.csv")
    print("calibration_summary: outputs/results/stage5_4_calibration_summary.txt")
    print("closed_loop_results: outputs/results/stage4_toy_hard_weighted_closed_loop_results.csv")
    print("hard_baseline_results: outputs/results/stage5_hard_baseline_results.csv")
    print("package: outputs/prism_stage5_6_weighted_results.zip")


if __name__ == "__main__":
    main()
