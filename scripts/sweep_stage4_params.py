from __future__ import annotations

import argparse
import copy
import csv
from itertools import product
from pathlib import Path

from prism.config import load_config
from prism.scripts.run_closed_loop import build_model, load_checkpoint_if_available, resolve_device, run_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep PRISM Stage-4 closed-loop stabilization parameters.")
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"), help="Path to YAML config.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("prism/outputs/checkpoints/best.pt"),
        help="Path to PRISM checkpoint.",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per parameter setting.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/results/stage4_param_sweep.csv"),
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

    base_config = load_config(args.config)
    # Fast sweep: keep the closed-loop planner intact, but use one MC sample per step
    # so a 4x3x3 grid remains practical to run locally.
    base_config["num_mc_samples"] = 1
    base_seed = int(base_config.get("data", {}).get("seed", 42))
    device = resolve_device(args.device)
    model = build_model(base_config).to(device)
    load_checkpoint_if_available(model, args.checkpoint, device)

    delta_values = [0.8, 1.0, 1.2, 1.5]
    goal_weight_values = [0.1, 0.3, 0.5]
    max_step_size_values = [1.0, 2.0, 3.0]
    rows = []

    for sweep_idx, (delta, goal_weight, max_step_size) in enumerate(
        product(delta_values, goal_weight_values, max_step_size_values)
    ):
        config = copy.deepcopy(base_config)
        config["risk_threshold_delta"] = delta
        config["goal_weight"] = goal_weight
        config.setdefault("env", {})
        config["env"]["max_step_size"] = max_step_size

        successes = 0
        collisions = 0
        cumulative_risks = []
        path_lengths = []
        steps = []
        for episode_idx in range(args.episodes):
            metrics = run_episode(
                config=config,
                checkpoint_path=args.checkpoint,
                device=device,
                seed=base_seed + sweep_idx * 10_000 + episode_idx,
                save_visualization=False,
                verbose=False,
                model=model,
            )
            successes += int(metrics["success"])
            collisions += int(metrics["collision"])
            cumulative_risks.append(float(metrics["cumulative_risk"]))
            path_lengths.append(float(metrics["path_length"]))
            steps.append(float(metrics["total_steps"]))

        row = {
            "risk_threshold_delta": delta,
            "goal_weight": goal_weight,
            "max_step_size": max_step_size,
            "episodes": args.episodes,
            "success_rate": successes / args.episodes,
            "collision_rate": collisions / args.episodes,
            "avg_cumulative_risk": sum(cumulative_risks) / args.episodes,
            "avg_path_length": sum(path_lengths) / args.episodes,
            "avg_steps": sum(steps) / args.episodes,
        }
        rows.append(row)
        print(
            f"delta={delta:.1f} goal_weight={goal_weight:.1f} max_step_size={max_step_size:.1f} "
            f"success_rate={row['success_rate']:.3f} collision_rate={row['collision_rate']:.3f} "
            f"avg_cumulative_risk={row['avg_cumulative_risk']:.4f}",
            flush=True,
        )

    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "risk_threshold_delta",
                "goal_weight",
                "max_step_size",
                "episodes",
                "success_rate",
                "collision_rate",
                "avg_cumulative_risk",
                "avg_path_length",
                "avg_steps",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    top_rows = sorted(
        rows,
        key=lambda row: (-float(row["success_rate"]), float(row["collision_rate"]), float(row["avg_cumulative_risk"])),
    )[:5]
    print("Top 5 parameter settings:")
    for rank, row in enumerate(top_rows, start=1):
        print(
            f"{rank}. delta={row['risk_threshold_delta']:.1f}, "
            f"goal_weight={row['goal_weight']:.1f}, max_step_size={row['max_step_size']:.1f}, "
            f"success_rate={row['success_rate']:.3f}, collision_rate={row['collision_rate']:.3f}, "
            f"avg_cumulative_risk={row['avg_cumulative_risk']:.4f}",
            flush=True,
        )
    print(f"Saved Stage-4 parameter sweep to: {output_path}")


if __name__ == "__main__":
    main()
