from __future__ import annotations

import argparse
import copy
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
from prism.scripts.evaluate_baselines import build_loaded_model
from prism.scripts.evaluate_baselines_hard import build_hard_env
from prism.scripts.run_closed_loop import resolve_device
from prism.utils.uncertainty import compute_safe_risk, mc_dropout_predict, propagate_uncertainty


CALIBRATION_FIELDS = [
    "episode",
    "seed",
    "horizon",
    "mu_mae",
    "mu_rmse",
    "safe_risk_mae",
    "safe_risk_rmse",
    "mu_corr",
    "safe_risk_corr",
    "mu_mean",
    "mu_max",
    "mu_p95",
    "sigma_mean",
    "sigma_max",
    "sigma_p95",
    "sigma_error_corr",
    "safe_risk_mean",
    "safe_risk_max",
    "safe_risk_p95",
    "true_risk_mean",
    "true_risk_max",
    "true_risk_p95",
    "high_risk_precision",
    "high_risk_recall",
    "high_risk_f1",
    "precision_05",
    "recall_05",
    "f1_05",
    "precision_07",
    "recall_07",
    "f1_07",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose PRISM predicted-risk calibration on hard ToyFireEnv.")
    parser.add_argument("--config", type=Path, default=Path("configs/toy_train_hard_level_b.yaml"))
    parser.add_argument("--hard-config", type=Path, default=Path("configs/toy_eval_hard.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints_toy_hard_b/best.pt"))
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/results/stage5_4_prediction_calibration.csv"),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Device to use for model inference.",
    )
    return parser.parse_args()


def tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
    values = tensor.detach().cpu().float().flatten()
    if values.numel() == 0:
        return {"mean": float("nan"), "max": float("nan"), "p95": float("nan")}
    return {
        "mean": float(values.mean().item()),
        "max": float(values.max().item()),
        "p95": float(torch.quantile(values, 0.95).item()),
    }


def pearson_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.detach().cpu().float().flatten()
    b_flat = b.detach().cpu().float().flatten()
    if a_flat.numel() == 0 or b_flat.numel() == 0:
        return float("nan")
    a_centered = a_flat - a_flat.mean()
    b_centered = b_flat - b_flat.mean()
    denom = torch.linalg.norm(a_centered) * torch.linalg.norm(b_centered)
    if float(denom.item()) <= 1e-12:
        return float("nan")
    return float((a_centered * b_centered).sum().div(denom).item())


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt((pred.detach().cpu().float() - target.detach().cpu().float()).pow(2).mean()).item())


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float((pred.detach().cpu().float() - target.detach().cpu().float()).abs().mean().item())


def high_risk_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float) -> dict[str, float]:
    pred_mask = pred.detach().cpu().float() >= threshold
    target_mask = target.detach().cpu().float() >= threshold
    tp = int((pred_mask & target_mask).sum().item())
    fp = int((pred_mask & ~target_mask).sum().item())
    fn = int((~pred_mask & target_mask).sum().item())
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if math.isnan(precision) or math.isnan(recall) or (precision + recall) <= 0.0:
        f1 = float("nan")
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def true_future_risk_rollout(env: Any, horizon: int) -> torch.Tensor:
    rollout_env = copy.deepcopy(env)
    maps: list[torch.Tensor] = []
    for _ in range(horizon):
        rollout_env.timestep += 1
        rollout_env.update_dynamics()
        maps.append(rollout_env.risk_map.detach().clone().float())
    return torch.stack(maps, dim=0).unsqueeze(1)


def predict_risk_maps(
    model: torch.nn.Module,
    observation: dict[str, torch.Tensor],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    obs = observation["obs"].unsqueeze(0).to(device)
    with torch.no_grad():
        mu_mean, sigma_mc, mu_samples = mc_dropout_predict(
            model,
            obs,
            num_samples=int(config.get("num_mc_samples", 5)),
        )
        sigma_prop, _ = propagate_uncertainty(mu_samples, alpha=float(config.get("uncertainty_alpha", 0.7)))
        safe_risk = compute_safe_risk(mu_mean, sigma_prop, lambda_u=float(config.get("lambda_u", 0.5)))
    return {
        "mu": mu_mean[0].detach().cpu().float(),
        "sigma_mc": sigma_mc[0].detach().cpu().float(),
        "sigma_prop": sigma_prop[0].detach().cpu().float(),
        "safe_risk": safe_risk[0].detach().cpu().float(),
    }


def calibration_row(
    *,
    episode: int,
    seed: int,
    horizon_idx: int,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    safe_risk: torch.Tensor,
    true_risk: torch.Tensor,
) -> dict[str, Any]:
    abs_error = (mu - true_risk).abs()
    mu_stats = tensor_stats(mu)
    sigma_stats = tensor_stats(sigma)
    safe_stats = tensor_stats(safe_risk)
    true_stats = tensor_stats(true_risk)
    metrics_05 = high_risk_metrics(safe_risk, true_risk, threshold=0.5)
    metrics_07 = high_risk_metrics(safe_risk, true_risk, threshold=0.7)

    return {
        "episode": episode,
        "seed": seed,
        "horizon": horizon_idx + 1,
        "mu_mae": mae(mu, true_risk),
        "mu_rmse": rmse(mu, true_risk),
        "safe_risk_mae": mae(safe_risk, true_risk),
        "safe_risk_rmse": rmse(safe_risk, true_risk),
        "mu_corr": pearson_corr(mu, true_risk),
        "safe_risk_corr": pearson_corr(safe_risk, true_risk),
        "mu_mean": mu_stats["mean"],
        "mu_max": mu_stats["max"],
        "mu_p95": mu_stats["p95"],
        "sigma_mean": sigma_stats["mean"],
        "sigma_max": sigma_stats["max"],
        "sigma_p95": sigma_stats["p95"],
        "sigma_error_corr": pearson_corr(sigma, abs_error),
        "safe_risk_mean": safe_stats["mean"],
        "safe_risk_max": safe_stats["max"],
        "safe_risk_p95": safe_stats["p95"],
        "true_risk_mean": true_stats["mean"],
        "true_risk_max": true_stats["max"],
        "true_risk_p95": true_stats["p95"],
        "high_risk_precision": metrics_05["precision"],
        "high_risk_recall": metrics_05["recall"],
        "high_risk_f1": metrics_05["f1"],
        "precision_05": metrics_05["precision"],
        "recall_05": metrics_05["recall"],
        "f1_05": metrics_05["f1"],
        "precision_07": metrics_07["precision"],
        "recall_07": metrics_07["recall"],
        "f1_07": metrics_07["f1"],
    }


def run_calibration_episode(
    *,
    config: dict[str, Any],
    hard: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    episode: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    env = build_hard_env(config, hard, seed=seed)
    observation = env.reset(seed=seed)
    horizon = int(config.get("horizon", 5))
    pred = predict_risk_maps(model, observation, config, device)
    true_future = true_future_risk_rollout(env, horizon)

    rows = []
    for horizon_idx in range(horizon):
        rows.append(
            calibration_row(
                episode=episode,
                seed=seed,
                horizon_idx=horizon_idx,
                mu=pred["mu"][horizon_idx],
                sigma=pred["sigma_prop"][horizon_idx],
                safe_risk=pred["safe_risk"][horizon_idx],
                true_risk=true_future[horizon_idx],
            )
        )
    return rows, {**pred, "true_future": true_future, "current_risk": observation["risk_map"].view(1, 1, *observation["risk_map"].shape)}


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
        rows, _ = run_calibration_episode(
            config=config,
            hard=hard,
            model=model,
            device=device,
            episode=episode,
            seed=seed,
        )
        all_rows.extend(rows)
        avg_mu_mae = sum(float(row["mu_mae"]) for row in rows) / len(rows)
        avg_safe_mae = sum(float(row["safe_risk_mae"]) for row in rows) / len(rows)
        avg_recall = [float(row["recall_05"]) for row in rows if not math.isnan(float(row["recall_05"]))]
        recall_text = "nan" if not avg_recall else f"{sum(avg_recall) / len(avg_recall):.4f}"
        print(
            f"episode={episode + 1}/{args.episodes} seed={seed} "
            f"avg_mu_mae={avg_mu_mae:.6f} avg_safe_mae={avg_safe_mae:.6f} "
            f"avg_safe_recall_05={recall_text}"
        )

    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CALIBRATION_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved Stage-5.4 prediction calibration to: {output}")


if __name__ == "__main__":
    main()
