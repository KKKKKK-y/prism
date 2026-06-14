from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = PROJECT_ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from prism.scripts.diagnose_hard_planner import load_stage5_2_config
from prism.scripts.diagnose_prediction_calibration import (
    calibration_row,
    predict_risk_maps,
    true_future_risk_rollout,
)
from prism.scripts.evaluate_baselines import build_loaded_model
from prism.scripts.evaluate_baselines_hard import build_hard_env
from prism.scripts.run_closed_loop import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Stage-5.4 prediction calibration maps.")
    parser.add_argument("--config", type=Path, default=Path("configs/toy_train_hard_level_b.yaml"))
    parser.add_argument("--hard-config", type=Path, default=Path("configs/toy_eval_hard.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints_toy_hard_b/best.pt"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/visualizations/stage5_4_prediction_calibration_seed0.png"),
    )
    parser.add_argument(
        "--failure-output",
        type=Path,
        default=Path("outputs/visualizations/stage5_4_prediction_calibration_failure.png"),
    )
    parser.add_argument("--search-limit", type=int, default=80)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Device to use for model inference.",
    )
    return parser.parse_args()


def episode_prediction_bundle(
    *,
    config: dict[str, Any],
    hard: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    seed: int,
) -> dict[str, torch.Tensor]:
    env = build_hard_env(config, hard, seed=seed)
    observation = env.reset(seed=seed)
    horizon = int(config.get("horizon", 5))
    pred = predict_risk_maps(model, observation, config, device)
    true_future = true_future_risk_rollout(env, horizon)
    return {**pred, "true_future": true_future}


def _squeeze_map(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach().cpu().float()
    while value.ndim > 2:
        value = value[0]
    return value


def _mean_recall_05(bundle: dict[str, torch.Tensor], seed: int) -> float:
    recalls: list[float] = []
    horizon = int(bundle["true_future"].shape[0])
    for idx in range(horizon):
        row = calibration_row(
            episode=0,
            seed=seed,
            horizon_idx=idx,
            mu=bundle["mu"][idx],
            sigma=bundle["sigma_prop"][idx],
            safe_risk=bundle["safe_risk"][idx],
            true_risk=bundle["true_future"][idx],
        )
        value = float(row["recall_05"])
        if not math.isnan(value):
            recalls.append(value)
    if not recalls:
        return float("inf")
    return sum(recalls) / len(recalls)


def find_failure_seed(
    *,
    config: dict[str, Any],
    hard: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    start_seed: int,
    search_limit: int,
) -> tuple[int, dict[str, torch.Tensor], float]:
    best_seed = start_seed
    best_bundle = episode_prediction_bundle(config=config, hard=hard, model=model, device=device, seed=start_seed)
    best_recall = _mean_recall_05(best_bundle, start_seed)
    for seed in range(start_seed + 1, start_seed + search_limit):
        bundle = episode_prediction_bundle(config=config, hard=hard, model=model, device=device, seed=seed)
        recall = _mean_recall_05(bundle, seed)
        if recall < best_recall:
            best_seed = seed
            best_bundle = bundle
            best_recall = recall
    return best_seed, best_bundle, best_recall


def save_calibration_grid(bundle: dict[str, torch.Tensor], seed: int, output: Path) -> None:
    true_future = bundle["true_future"]
    mu = bundle["mu"]
    sigma = bundle["sigma_prop"]
    safe = bundle["safe_risk"]
    horizon = int(true_future.shape[0])
    rows = [
        ("true future risk", true_future, 0.0, 1.0, "inferno"),
        ("predicted mu", mu, 0.0, 1.0, "inferno"),
        ("sigma prop", sigma, 0.0, max(0.05, float(sigma.max().item())), "viridis"),
        ("safe risk", safe, 0.0, 1.0, "inferno"),
        ("abs error", (mu - true_future).abs(), 0.0, max(0.05, float((mu - true_future).abs().max().item())), "magma"),
    ]

    fig, axes = plt.subplots(len(rows), horizon, figsize=(3.2 * horizon, 3.1 * len(rows)), constrained_layout=True)
    for col in range(horizon):
        metric_row = calibration_row(
            episode=0,
            seed=seed,
            horizon_idx=col,
            mu=mu[col],
            sigma=sigma[col],
            safe_risk=safe[col],
            true_risk=true_future[col],
        )
        for row_idx, (label, tensor, vmin, vmax, cmap) in enumerate(rows):
            ax = axes[row_idx, col]
            image = _squeeze_map(tensor[col])
            ax.imshow(image.numpy(), cmap=cmap, origin="upper", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label, fontsize=10)
            if row_idx == 0:
                recall = metric_row["recall_05"]
                recall_text = "nan" if math.isnan(float(recall)) else f"{float(recall):.2f}"
                ax.set_title(
                    f"h={col + 1}\nMAE={float(metric_row['mu_mae']):.3f} "
                    f"RMSE={float(metric_row['mu_rmse']):.3f}\nrecall={recall_text}",
                    fontsize=9,
                )

    fig.suptitle(f"Stage-5.4 prediction calibration, seed={seed}", fontsize=14)
    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config, hard = load_stage5_2_config(args.config, args.hard_config)
    device = resolve_device(args.device)
    model = build_loaded_model(config, args.checkpoint, device)

    seed_bundle = episode_prediction_bundle(config=config, hard=hard, model=model, device=device, seed=args.seed)
    save_calibration_grid(seed_bundle, args.seed, args.output)
    print(f"Saved Stage-5.4 prediction calibration visualization to: {args.output}")

    failure_seed, failure_bundle, recall = find_failure_seed(
        config=config,
        hard=hard,
        model=model,
        device=device,
        start_seed=args.seed,
        search_limit=args.search_limit,
    )
    save_calibration_grid(failure_bundle, failure_seed, args.failure_output)
    recall_text = "nan" if math.isnan(float(recall)) else f"{recall:.4f}"
    print(
        f"Saved Stage-5.4 prediction calibration failure visualization to: {args.failure_output} "
        f"(seed={failure_seed}, avg_recall_05={recall_text})"
    )


if __name__ == "__main__":
    main()
