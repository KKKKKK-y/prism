from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = PROJECT_ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from prism.scripts.diagnose_hard_planner import load_stage5_2_config
from prism.scripts.diagnose_prediction_calibration import (
    high_risk_metrics,
    mae,
    pearson_corr,
    predict_risk_maps,
    true_future_risk_rollout,
)
from prism.scripts.evaluate_baselines import build_loaded_model
from prism.scripts.evaluate_baselines_hard import build_hard_env
from prism.scripts.run_closed_loop import resolve_device


COMPARE_FIELDS = [
    "episode",
    "seed",
    "horizon",
    "threshold",
    "current_mae",
    "mu_mae",
    "safe_risk_mae",
    "current_corr",
    "mu_corr",
    "safe_risk_corr",
    "current_high_risk_recall",
    "mu_high_risk_recall",
    "safe_risk_high_risk_recall",
    "current_high_risk_precision",
    "mu_high_risk_precision",
    "safe_risk_high_risk_precision",
    "true_risk_mean",
    "true_risk_max",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare current risk and PRISM predictions against true future risk.")
    parser.add_argument("--config", type=Path, default=Path("configs/toy_train_hard_level_b.yaml"))
    parser.add_argument("--hard-config", type=Path, default=Path("configs/toy_eval_hard.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints_toy_hard_b/best.pt"))
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/results/stage5_4_current_vs_predicted.csv"),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Device to use for model inference.",
    )
    return parser.parse_args()


def compare_episode(
    *,
    config: dict[str, Any],
    hard: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    episode: int,
    seed: int,
    threshold: float,
) -> list[dict[str, Any]]:
    env = build_hard_env(config, hard, seed=seed)
    observation = env.reset(seed=seed)
    horizon = int(config.get("horizon", 5))
    pred = predict_risk_maps(model, observation, config, device)
    true_future = true_future_risk_rollout(env, horizon)
    current_risk = observation["risk_map"].view(1, *observation["risk_map"].shape).float()

    rows: list[dict[str, Any]] = []
    for horizon_idx in range(horizon):
        true_risk = true_future[horizon_idx]
        current_map = current_risk
        mu = pred["mu"][horizon_idx]
        safe = pred["safe_risk"][horizon_idx]
        current_metrics = high_risk_metrics(current_map, true_risk, threshold)
        mu_metrics = high_risk_metrics(mu, true_risk, threshold)
        safe_metrics = high_risk_metrics(safe, true_risk, threshold)
        rows.append(
            {
                "episode": episode,
                "seed": seed,
                "horizon": horizon_idx + 1,
                "threshold": threshold,
                "current_mae": mae(current_map, true_risk),
                "mu_mae": mae(mu, true_risk),
                "safe_risk_mae": mae(safe, true_risk),
                "current_corr": pearson_corr(current_map, true_risk),
                "mu_corr": pearson_corr(mu, true_risk),
                "safe_risk_corr": pearson_corr(safe, true_risk),
                "current_high_risk_recall": current_metrics["recall"],
                "mu_high_risk_recall": mu_metrics["recall"],
                "safe_risk_high_risk_recall": safe_metrics["recall"],
                "current_high_risk_precision": current_metrics["precision"],
                "mu_high_risk_precision": mu_metrics["precision"],
                "safe_risk_high_risk_precision": safe_metrics["precision"],
                "true_risk_mean": float(true_risk.mean().item()),
                "true_risk_max": float(true_risk.max().item()),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError(f"--episodes must be positive, got {args.episodes}")
    config, hard = load_stage5_2_config(args.config, args.hard_config)
    device = resolve_device(args.device)
    model = build_loaded_model(config, args.checkpoint, device)
    base_seed = int(config.get("data", {}).get("seed", config.get("seed", 42)))

    all_rows: list[dict[str, Any]] = []
    for episode in range(args.episodes):
        seed = base_seed + episode
        rows = compare_episode(
            config=config,
            hard=hard,
            model=model,
            device=device,
            episode=episode,
            seed=seed,
            threshold=args.threshold,
        )
        all_rows.extend(rows)
        current_mae = sum(float(row["current_mae"]) for row in rows) / len(rows)
        mu_mae = sum(float(row["mu_mae"]) for row in rows) / len(rows)
        safe_mae = sum(float(row["safe_risk_mae"]) for row in rows) / len(rows)
        safe_recalls = [float(row["safe_risk_high_risk_recall"]) for row in rows if not math.isnan(float(row["safe_risk_high_risk_recall"]))]
        safe_recall = "nan" if not safe_recalls else f"{sum(safe_recalls) / len(safe_recalls):.4f}"
        print(
            f"episode={episode + 1}/{args.episodes} seed={seed} "
            f"current_mae={current_mae:.6f} mu_mae={mu_mae:.6f} "
            f"safe_mae={safe_mae:.6f} safe_recall={safe_recall}"
        )

    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COMPARE_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved Stage-5.4 current-vs-predicted comparison to: {output}")


if __name__ == "__main__":
    main()
