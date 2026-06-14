from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prism.config import load_config
from prism.envs import ToyFireEnv
from prism.models import RiskPredictor
from prism.planners import sample_candidate_trajectories, select_safe_trajectory
from prism.trainers.trainer import select_device
from prism.utils.uncertainty import compute_safe_risk, mc_dropout_predict, propagate_uncertainty


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PRISM closed-loop evaluation in ToyFireEnv.")
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"), help="Path to YAML config.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("prism/outputs/checkpoints/best.pt"),
        help="Path to PRISM checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/visualizations/stage4_closed_loop.png"),
        help="Output visualization path.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Device to use for model inference.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-step closed-loop debug metrics.")
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


def build_env(config: dict[str, Any], seed: int | None = None) -> ToyFireEnv:
    env_cfg = config.get("env", {})
    return ToyFireEnv(
        map_size=int(env_cfg.get("map_size", config.get("image_size", 64))),
        obs_window=int(config.get("obs_window", 4)),
        channels=int(config.get("channels", 5)),
        max_steps=int(env_cfg.get("max_steps", 100)),
        num_fire_sources=int(env_cfg.get("num_fire_sources", 2)),
        obstacle_density=float(env_cfg.get("obstacle_density", 0.05)),
        fire_spread_rate=float(env_cfg.get("fire_spread_rate", 0.08)),
        smoke_spread_rate=float(env_cfg.get("smoke_spread_rate", 0.12)),
        risk_threshold_collision=float(env_cfg.get("risk_threshold_collision", 0.85)),
        goal_radius=float(env_cfg.get("goal_radius", 3.0)),
        max_step_size=float(env_cfg.get("max_step_size", 2.0)),
        seed=seed,
    )


def load_checkpoint_if_available(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint_path = checkpoint_path.expanduser()
    if not checkpoint_path.exists():
        print(f"Warning: checkpoint not found at {checkpoint_path}. Using randomly initialized model.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint from: {checkpoint_path}")


def infer_safe_risk(model: torch.nn.Module, obs: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    """
    obs: [1, k, C, 64, 64]
    safe_risk: [1, H, 1, 64, 64]
    """
    with torch.no_grad():
        mu_mean, _, mu_samples = mc_dropout_predict(
            model,
            obs,
            num_samples=int(config.get("num_mc_samples", 5)),
        )
        sigma_prop, _ = propagate_uncertainty(mu_samples, alpha=float(config.get("uncertainty_alpha", 0.7)))
        safe_risk = compute_safe_risk(mu_mean, sigma_prop, lambda_u=float(config.get("lambda_u", 0.5)))
    return safe_risk


def compute_path_length(path: list[torch.Tensor]) -> float:
    if len(path) < 2:
        return 0.0
    total = 0.0
    for prev, cur in zip(path[:-1], path[1:]):
        total += float(torch.linalg.norm(cur - prev).item())
    return total


def run_episode(
    config: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
    seed: int | None = None,
    save_visualization: bool = False,
    visualization_path: Path | None = None,
    verbose: bool = False,
    model: torch.nn.Module | None = None,
) -> dict[str, Any]:
    env = build_env(config, seed=seed)
    observation = env.reset(seed=seed)
    if model is None:
        model = build_model(config).to(device)
        load_checkpoint_if_available(model, checkpoint_path, device)
    else:
        model = model.to(device)
    model.eval()

    horizon = int(config.get("horizon", 5))
    map_size = int(config.get("env", {}).get("map_size", config.get("image_size", 64)))
    num_trajectories = int(config.get("num_trajectories", 64))
    delta = float(config.get("risk_threshold_delta", 0.8))
    goal_weight = float(config.get("goal_weight", 0.1))
    noise_scale = float(config.get("trajectory_noise_scale", 4.0))
    progress_weight = float(config.get("progress_weight", 0.2))
    backtrack_penalty = float(config.get("backtrack_penalty", 1.0))
    max_step_size = float(config.get("env", {}).get("max_step_size", 2.0))

    path = [observation["robot_xy"].clone()]
    cumulative_risk = 0.0
    done = False
    info: dict[str, Any] = {"success": False, "collision": False, "timeout": False, "risk_at_robot": 0.0}

    while not done:
        # env obs: [k, C, 64, 64], model obs: [1, k, C, 64, 64]
        obs = observation["obs"].unsqueeze(0).to(device)
        safe_risk = infer_safe_risk(model, obs, config)
        # planner safe_risk_map: [H, 1, 64, 64]
        safe_risk_map = safe_risk[0].detach().cpu()

        robot_xy = observation["robot_xy"]
        goal_xy = observation["goal_xy"]
        # trajectories: [N, H, 2], coordinates are [x, y].
        trajectories = sample_candidate_trajectories(
            start_xy=robot_xy,
            goal_xy=goal_xy,
            num_trajectories=num_trajectories,
            horizon=horizon,
            map_size=map_size,
            noise_scale=noise_scale,
            max_step_size=max_step_size,
        )
        best_traj, _, _, _, planner_info = select_safe_trajectory(
            safe_risk_map=safe_risk_map,
            trajectories=trajectories,
            goal_xy=goal_xy,
            start_xy=robot_xy,
            delta=delta,
            goal_weight=goal_weight,
            progress_weight=progress_weight,
            backtrack_penalty=backtrack_penalty,
        )
        action_xy = best_traj[0]
        observation, done, info = env.step(action_xy)
        path.append(observation["robot_xy"].clone())
        cumulative_risk += float(info["risk_at_robot"])

        if verbose:
            print(
                f"step={info['timestep']} "
                f"robot_xy=({observation['robot_xy'][0].item():.2f},{observation['robot_xy'][1].item():.2f}) "
                f"distance_to_goal={info['distance_to_goal']:.3f} "
                f"current_risk={info['risk_at_robot']:.4f} "
                f"best_risk={planner_info['best_risk']:.4f} "
                f"num_feasible={planner_info['num_feasible']} "
                f"action_distance={info['action_distance']:.3f} "
                f"collision={info['collision']} done={done}"
            )

    path_length = float(info.get("path_length", compute_path_length(path)))
    total_steps = int(info["timestep"])
    average_risk = cumulative_risk / max(1, total_steps)
    metrics = {
        "success": bool(info["success"]),
        "collision": bool(info["collision"]),
        "obstacle_collision": bool(info.get("obstacle_collision", False)),
        "high_risk_collision": bool(info.get("high_risk_collision", False)),
        "timeout": bool(info["timeout"]),
        "failure_reason": str(info.get("failure_reason", "none")),
        "total_steps": total_steps,
        "cumulative_risk": cumulative_risk,
        "path_length": path_length,
        "average_risk": average_risk,
        "path": torch.stack(path, dim=0),
        "final_risk_map": observation["risk_map"].clone(),
        "start_xy": env.start_xy.clone(),
        "goal_xy": env.goal_xy.clone(),
        "fire_sources": list(env.fire_sources),
    }

    if save_visualization and visualization_path is not None:
        save_closed_loop_visualization(metrics, visualization_path)

    return metrics


def save_closed_loop_visualization(metrics: dict[str, Any], output_path: Path) -> None:
    risk_map = metrics["final_risk_map"].detach().cpu()
    path = metrics["path"].detach().cpu()
    start_xy = metrics["start_xy"].detach().cpu()
    goal_xy = metrics["goal_xy"].detach().cpu()

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    im = ax.imshow(risk_map.numpy(), cmap="inferno", origin="upper")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.plot(path[:, 0].numpy(), path[:, 1].numpy(), color="#38bdf8", linewidth=2.8, label="Robot path")
    ax.scatter([start_xy[0].item()], [start_xy[1].item()], marker="o", s=90, color="#22c55e", edgecolor="white", label="Start")
    ax.scatter([goal_xy[0].item()], [goal_xy[1].item()], marker="*", s=150, color="#facc15", edgecolor="black", label="Goal")
    for fire_x, fire_y in metrics["fire_sources"]:
        ax.scatter([fire_x], [fire_y], marker="x", s=80, color="#ef4444", linewidths=2.2)

    ax.set_xlim(0, risk_map.shape[1] - 1)
    ax.set_ylim(risk_map.shape[0] - 1, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("x / image column")
    ax.set_ylabel("y / image row")
    ax.set_title(
        "PRISM Stage-4 Closed-Loop Toy Fire Evaluation\n"
        f"success={metrics['success']} collision={metrics['collision']} "
        f"cumulative_risk={metrics['cumulative_risk']:.4f} steps={metrics['total_steps']}"
    )
    ax.legend(loc="lower left", framealpha=0.85)
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(args.device)
    metrics = run_episode(
        config=config,
        checkpoint_path=args.checkpoint,
        device=device,
        seed=int(config.get("data", {}).get("seed", 42)),
        save_visualization=True,
        visualization_path=args.output,
        verbose=args.verbose,
    )
    print(f"success: {metrics['success']}")
    print(f"collision: {metrics['collision']}")
    print(f"failure_reason: {metrics['failure_reason']}")
    print(f"total_steps: {metrics['total_steps']}")
    print(f"cumulative_risk: {metrics['cumulative_risk']:.6f}")
    print(f"path_length: {metrics['path_length']:.6f}")
    print(f"average_risk: {metrics['average_risk']:.6f}")
    print(f"Saved closed-loop visualization to: {args.output}")


if __name__ == "__main__":
    main()
