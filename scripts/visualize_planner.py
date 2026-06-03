from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from prism.config import load_config
from prism.datasets import DummyFireRiskDataset
from prism.models import RiskPredictor
from prism.planners import sample_candidate_trajectories, select_safe_trajectory
from prism.trainers.trainer import select_device
from prism.utils.uncertainty import compute_safe_risk, mc_dropout_predict, propagate_uncertainty


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize PRISM Stage-3 trajectory safe-risk planning.")
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"), help="Path to YAML config.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("prism/outputs/checkpoints/best.pt"),
        help="Path to Stage-1/2 checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("prism/outputs/visualizations/stage3_planner.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Device to use for Stage-2 safe-risk inference.",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return select_device()
    if name == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def build_model(config: dict[str, Any]) -> RiskPredictor:
    return RiskPredictor(
        in_channels=int(config.get("channels", 6)),
        obs_window=int(config.get("obs_window", 4)),
        horizon=int(config.get("horizon", 5)),
        base_channels=int(config.get("base_channels", 32)),
        latent_channels=int(config.get("latent_channels", 128)),
        gru_layers=int(config.get("gru_layers", 1)),
        dropout_p=float(config.get("dropout_p", 0.1)),
    )


def build_dummy_safe_risk_map(horizon: int, image_size: int) -> torch.Tensor:
    """
    safe_risk_map: [H, 1, 64, 64]
    Image/map indexing is [y, x], where y is row and x is column.
    """
    yy, xx = torch.meshgrid(
        torch.arange(image_size, dtype=torch.float32),
        torch.arange(image_size, dtype=torch.float32),
        indexing="ij",
    )
    base = torch.full((image_size, image_size), 0.02, dtype=torch.float32)
    obstacle = 0.35 * torch.exp(-((xx - 31.0).pow(2) + (yy - 34.0).pow(2)) / (2.0 * 8.0**2))
    risk = (base + obstacle).clamp(0.0, 1.0)
    horizon_scale = torch.linspace(0.85, 1.15, horizon).view(horizon, 1, 1)
    return (risk.unsqueeze(0) * horizon_scale).unsqueeze(1).clamp(0.0, 1.0)


def load_stage2_safe_risk(config: dict[str, Any], checkpoint_path: Path, device: torch.device) -> torch.Tensor:
    checkpoint_path = checkpoint_path.expanduser()
    horizon = int(config.get("horizon", 5))
    image_size = int(config.get("image_size", 64))
    if not checkpoint_path.exists():
        print(f"Warning: checkpoint not found at {checkpoint_path}. Using dummy safe-risk map.")
        return build_dummy_safe_risk_map(horizon, image_size)

    dataset = DummyFireRiskDataset(
        num_samples=1,
        obs_window=int(config.get("obs_window", 4)),
        horizon=horizon,
        channels=int(config.get("channels", 6)),
        image_size=image_size,
        seed=int(config.get("data", {}).get("seed", 42)),
    )
    sample = dataset[0]
    # obs: [1, k, C, 64, 64]
    obs = sample["obs"].unsqueeze(0).to(device)
    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint from: {checkpoint_path}")

    with torch.no_grad():
        # mu_mean: [1, H, 1, 64, 64], mu_samples: [N, 1, H, 1, 64, 64]
        mu_mean, _, mu_samples = mc_dropout_predict(
            model,
            obs,
            num_samples=int(config.get("num_mc_samples", 5)),
        )
        # sigma_prop: [1, H, 1, 64, 64]
        sigma_prop, _ = propagate_uncertainty(mu_samples, alpha=float(config.get("uncertainty_alpha", 0.7)))
        # safe_risk: [1, H, 1, 64, 64]
        safe_risk = compute_safe_risk(mu_mean, sigma_prop, lambda_u=float(config.get("lambda_u", 0.5)))

    return safe_risk[0].detach().cpu()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    horizon = int(config.get("horizon", 5))
    image_size = int(config.get("image_size", 64))
    num_trajectories = int(config.get("num_trajectories", 64))
    delta = float(config.get("risk_threshold_delta", 0.8))
    goal_weight = float(config.get("goal_weight", 0.1))
    noise_scale = float(config.get("trajectory_noise_scale", 4.0))

    device = resolve_device(args.device)
    # safe_risk_map: [H, 1, 64, 64]
    safe_risk_map = load_stage2_safe_risk(config, args.checkpoint, device)
    start_xy = torch.tensor([5.0, 58.0])
    goal_xy = torch.tensor([56.0, 8.0])

    # trajectories: [N, H, 2], coordinates are [x, y].
    trajectories = sample_candidate_trajectories(
        start_xy=start_xy,
        goal_xy=goal_xy,
        num_trajectories=num_trajectories,
        horizon=horizon,
        map_size=image_size,
        noise_scale=noise_scale,
    )
    best_traj, _, trajectory_risks, feasible_mask, info = select_safe_trajectory(
        safe_risk_map=safe_risk_map,
        trajectories=trajectories,
        goal_xy=goal_xy,
        start_xy=start_xy,
        delta=delta,
        goal_weight=goal_weight,
        progress_weight=float(config.get("progress_weight", 0.2)),
        backtrack_penalty=float(config.get("backtrack_penalty", 1.0)),
    )

    background = safe_risk_map[-1, 0].detach().cpu()
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    im = ax.imshow(background.numpy(), cmap="inferno", origin="upper")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    trajectories_cpu = trajectories.detach().cpu()
    feasible_cpu = feasible_mask.detach().cpu()
    for idx, trajectory in enumerate(trajectories_cpu):
        x = trajectory[:, 0].numpy()
        y = trajectory[:, 1].numpy()
        if bool(feasible_cpu[idx]):
            ax.plot(x, y, color="#4ade80", linewidth=0.9, alpha=0.30, linestyle="-")
        else:
            ax.plot(x, y, color="#94a3b8", linewidth=0.8, alpha=0.22, linestyle="--")

    best_cpu = best_traj.detach().cpu()
    ax.plot(best_cpu[:, 0].numpy(), best_cpu[:, 1].numpy(), color="#38bdf8", linewidth=3.0, label="Best trajectory")
    ax.scatter([start_xy[0].item()], [start_xy[1].item()], marker="o", s=90, color="#22c55e", edgecolor="white", label="Start")
    ax.scatter([goal_xy[0].item()], [goal_xy[1].item()], marker="*", s=150, color="#facc15", edgecolor="black", label="Goal")
    ax.set_xlim(0, image_size - 1)
    ax.set_ylim(image_size - 1, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("x / image column")
    ax.set_ylabel("y / image row")
    ax.set_title(
        "PRISM Stage-3 Trajectory Safe-Risk Constraint\n"
        f"num_feasible={info['num_feasible']}  best_risk={info['best_risk']:.4f}  delta={delta:.4f}"
    )
    legend_handles = [
        Line2D([0], [0], color="#4ade80", lw=1.5, alpha=0.55, label="Feasible candidates"),
        Line2D([0], [0], color="#94a3b8", lw=1.5, alpha=0.55, linestyle="--", label="Rejected candidates"),
        Line2D([0], [0], color="#38bdf8", lw=3.0, label="Best trajectory"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#22c55e", markeredgecolor="white", markersize=9, label="Start"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#facc15", markeredgecolor="black", markersize=12, label="Goal"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", framealpha=0.85)

    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved planner visualization to: {output_path}")


if __name__ == "__main__":
    main()
