from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from prism.config import load_config
from prism.scripts.run_closed_loop import build_model, load_checkpoint_if_available, resolve_device, run_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PRISM closed-loop ToyFireEnv over multiple episodes.")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes to run.")
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
        default=Path("outputs/results/stage4_closed_loop_results.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Device to use for model inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError(f"--episodes must be positive, got {args.episodes}")

    config = load_config(args.config)
    base_seed = int(config.get("data", {}).get("seed", 42))
    device = resolve_device(args.device)
    model = build_model(config).to(device)
    load_checkpoint_if_available(model, args.checkpoint, device)
    rows = []
    failure_counts = {
        "reached_goal": 0,
        "collision_high_risk": 0,
        "collision_obstacle": 0,
        "timeout": 0,
    }

    for episode_idx in range(args.episodes):
        metrics = run_episode(
            config=config,
            checkpoint_path=args.checkpoint,
            device=device,
            seed=base_seed + episode_idx,
            save_visualization=False,
            verbose=False,
            model=model,
        )
        if metrics["success"]:
            failure_reason = "reached_goal"
        else:
            failure_reason = metrics["failure_reason"]
        if failure_reason in failure_counts:
            failure_counts[failure_reason] += 1
        rows.append(
            {
                "episode": episode_idx,
                "success": int(metrics["success"]),
                "collision": int(metrics["collision"]),
                "timeout": int(metrics["timeout"]),
                "failure_reason": failure_reason,
                "cumulative_risk": metrics["cumulative_risk"],
                "path_length": metrics["path_length"],
                "steps": metrics["total_steps"],
            }
        )
        print(
            f"episode={episode_idx + 1}/{args.episodes} "
            f"success={metrics['success']} collision={metrics['collision']} "
            f"steps={metrics['total_steps']} cumulative_risk={metrics['cumulative_risk']:.4f}"
        )

    success_rate = sum(row["success"] for row in rows) / args.episodes
    collision_rate = sum(row["collision"] for row in rows) / args.episodes
    timeout_rate = sum(row["timeout"] for row in rows) / args.episodes
    avg_cumulative_risk = sum(float(row["cumulative_risk"]) for row in rows) / args.episodes
    avg_path_length = sum(float(row["path_length"]) for row in rows) / args.episodes
    avg_steps = sum(float(row["steps"]) for row in rows) / args.episodes

    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "success",
                "collision",
                "timeout",
                "failure_reason",
                "cumulative_risk",
                "path_length",
                "steps",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Closed-loop evaluation summary:")
    print(f"success_rate: {success_rate:.6f}")
    print(f"collision_rate: {collision_rate:.6f}")
    print(f"timeout_rate: {timeout_rate:.6f}")
    print(f"reached_goal: {failure_counts['reached_goal']}")
    print(f"collision_high_risk: {failure_counts['collision_high_risk']}")
    print(f"collision_obstacle: {failure_counts['collision_obstacle']}")
    print(f"timeout: {failure_counts['timeout']}")
    print(f"avg_cumulative_risk: {avg_cumulative_risk:.6f}")
    print(f"avg_path_length: {avg_path_length:.6f}")
    print(f"avg_steps: {avg_steps:.6f}")
    print(f"Saved closed-loop results to: {output_path}")


if __name__ == "__main__":
    main()
