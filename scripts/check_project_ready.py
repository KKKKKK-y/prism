from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether PRISM is ready for Ubuntu training.")
    parser.add_argument("--root", type=Path, default=Path("."), help="PRISM project root.")
    return parser.parse_args()


def assert_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required path: {path}")


def assert_writable(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    probe = directory / ".write_test"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()

    required_dirs = [
        "configs",
        "datasets",
        "envs",
        "models",
        "planners",
        "trainers",
        "utils",
        "scripts",
    ]
    required_files = [
        "configs/smoke.yaml",
        "configs/toy_train.yaml",
        "configs/prism.yaml",
        "scripts/test_env.py",
        "scripts/train.py",
        "scripts/generate_toy_dataset.py",
        "scripts/evaluate_prediction_on_toy.py",
        "scripts/evaluate_closed_loop.py",
    ]
    writable_dirs = [
        "outputs",
        "outputs/results",
        "outputs/visualizations",
        "outputs/checkpoints_toy",
    ]

    for directory in required_dirs:
        assert_exists(root / directory)
    for file_path in required_files:
        assert_exists(root / file_path)
    for directory in writable_dirs:
        assert_writable(root / directory)

    from prism.datasets import DummyFireRiskDataset, ToyFireRiskDataset
    from prism.envs import ToyFireEnv
    from prism.models import RiskPredictor
    from prism.planners import sample_candidate_trajectories, select_safe_trajectory
    from prism.utils import mc_dropout_predict

    imported = [
        RiskPredictor,
        DummyFireRiskDataset,
        ToyFireRiskDataset,
        ToyFireEnv,
        mc_dropout_predict,
        sample_candidate_trajectories,
        select_safe_trajectory,
    ]
    if not all(imported):
        raise RuntimeError("One or more PRISM imports failed.")

    print("PRISM project ready for Ubuntu training.")


if __name__ == "__main__":
    main()
