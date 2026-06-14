from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = PROJECT_ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full PRISM Level-A reproduction from scratch.")
    parser.add_argument("--config", type=Path, default=Path("configs/reproduction_level_a.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None, help="Override evaluation episode count.")
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
    print("Full reproduction environment:")
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


def count_npz(path: Path) -> int:
    with np.load(path) as data:
        return int(data["obs"].shape[0])


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_final_config(config_path: Path, checkpoint_path: Path, config: dict[str, Any], episodes: int) -> Path:
    planner = config.get("planner", {}) if isinstance(config.get("planner"), dict) else {}
    path = PROJECT_ROOT / "outputs" / "results" / "full_reproduction_final_fusion.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "planner": {
            "final_method": "alpha_fusion",
            "alpha": float(planner.get("alpha", 0.4)),
            "lambda_u": float(config.get("lambda_u", 0.5)),
            "uncertainty_alpha": float(config.get("uncertainty_alpha", 0.7)),
            "num_mc_samples": int(config.get("num_mc_samples", 5)),
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
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return path


def write_summary(
    *,
    path: Path,
    config_path: Path,
    train_count: int,
    val_count: int,
    test_count: int,
    epochs: int,
    prediction_csv: Path,
    closed_loop_csv: Path,
    final_csv: Path,
    plot_path: Path,
) -> None:
    rows = read_csv(final_csv)
    current = next((row for row in rows if row.get("method") == "current_only"), None)
    ours = next((row for row in rows if row.get("method") == "alpha_fusion_0.4"), None)
    lines = [
        "PRISM full reproduction summary",
        "===============================",
        f"config: {config_path}",
        f"train_samples: {train_count}",
        f"val_samples: {val_count}",
        f"test_samples: {test_count}",
        f"epochs: {epochs}",
        f"prediction_metrics: {prediction_csv}",
        f"closed_loop_results: {closed_loop_csv}",
        f"main_table: {final_csv}",
        f"main_plot: {plot_path}",
    ]
    if current and ours:
        lines.extend(
            [
                "",
                "Final hard benchmark:",
                f"current_only_success: {float(current['success_rate']):.6f}",
                f"current_only_collision: {float(current['collision_rate']):.6f}",
                f"alpha_fusion_0.4_success: {float(ours['success_rate']):.6f}",
                f"alpha_fusion_0.4_collision: {float(ours['collision_rate']):.6f}",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = args.config
    config = load_yaml(config_path)
    seed = int(config.get("seed", config.get("data", {}).get("seed", 42)))
    set_seed(seed)
    print_environment()
    print(f"seed: {seed}")

    train_npz = PROJECT_ROOT / Path(config.get("train_npz", "outputs/datasets_reproduction_level_a/toy_fire_train.npz"))
    val_npz = PROJECT_ROOT / Path(config.get("val_npz", "outputs/datasets_reproduction_level_a/toy_fire_val.npz"))
    test_npz = PROJECT_ROOT / Path(config.get("test_npz", "outputs/datasets_reproduction_level_a/toy_fire_test.npz"))
    dataset_dir = train_npz.parent
    checkpoint_path = PROJECT_ROOT / Path(config.get("checkpoint_dir", "outputs/checkpoints_reproduction_level_a")) / "best.pt"
    results_dir = PROJECT_ROOT / Path(config.get("results_dir", "outputs/results"))
    vis_dir = PROJECT_ROOT / Path(config.get("visualization_dir", "outputs/visualizations"))
    epochs = int(args.epochs if args.epochs is not None else config.get("epochs", 50))
    episodes = int(args.episodes if args.episodes is not None else config.get("eval_episodes", 100))

    print("\nphase=dataset_generation", flush=True)
    run_command(
        [
            "scripts/generate_toy_dataset.py",
            "--config",
            str(config_path),
            "--train_episodes",
            str(int(config.get("train_episodes", 100))),
            "--val_episodes",
            str(int(config.get("val_episodes", 20))),
            "--test_episodes",
            str(int(config.get("test_episodes", 20))),
            "--output_dir",
            str(dataset_dir),
        ]
    )

    train_count = count_npz(train_npz)
    val_count = count_npz(val_npz)
    test_count = count_npz(test_npz)

    print("\nphase=training", flush=True)
    run_command(["scripts/train.py", "--config", str(config_path), "--epochs", str(epochs)])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Expected checkpoint was not created: {checkpoint_path}")

    prediction_csv = results_dir / "full_reproduction_prediction_metrics.csv"
    closed_loop_csv = results_dir / "full_reproduction_closed_loop_results.csv"
    final_csv = results_dir / "full_reproduction_main_table.csv"
    final_episode_csv = results_dir / "full_reproduction_episode_results.csv"
    plot_path = vis_dir / "full_reproduction_main_comparison.png"
    summary_path = results_dir / "full_reproduction_summary.txt"

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
            str(episodes),
            "--output",
            str(closed_loop_csv),
            "--device",
            args.device,
        ]
    )

    print("\nphase=hard_final_fusion_eval", flush=True)
    final_config_path = write_final_config(config_path, checkpoint_path, config, episodes)
    run_command(
        [
            "scripts/run_stage5_9_final_fusion_eval.py",
            "--config",
            str(final_config_path),
            "--episodes",
            str(episodes),
            "--summary-output",
            str(final_csv),
            "--episode-output",
            str(final_episode_csv),
            "--device",
            args.device,
        ]
    )

    print("\nphase=plot_main_comparison", flush=True)
    run_command(["scripts/plot_stage5_9_final_fusion.py", "--input", str(final_csv), "--output", str(plot_path)])

    write_summary(
        path=summary_path,
        config_path=config_path,
        train_count=train_count,
        val_count=val_count,
        test_count=test_count,
        epochs=epochs,
        prediction_csv=prediction_csv,
        closed_loop_csv=closed_loop_csv,
        final_csv=final_csv,
        plot_path=plot_path,
    )
    print(f"Saved full reproduction summary to: {summary_path}")


if __name__ == "__main__":
    main()
