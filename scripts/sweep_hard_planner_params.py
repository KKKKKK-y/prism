from __future__ import annotations

import argparse
import copy
import csv
import itertools
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = PROJECT_ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from prism.planners import STAGE5_METHODS
from prism.scripts.diagnose_hard_planner import load_stage5_2_config, run_diagnostic_episode
from prism.scripts.evaluate_baselines import MODEL_METHODS, build_loaded_model, needs_model
from prism.scripts.run_closed_loop import resolve_device


SWEEP_FIELDS = [
    "lambda_u",
    "risk_threshold_delta",
    "goal_weight",
    "trajectory_noise_scale",
    "num_trajectories",
    "episodes",
    "success_rate",
    "collision_rate",
    "timeout_rate",
    "avg_cumulative_risk",
    "avg_path_length",
    "avg_steps",
    "avg_feasible_ratio",
    "avg_selected_traj_risk",
    "avg_selected_final_distance",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep Stage-5.2 hard planner parameters without retraining.")
    parser.add_argument("--config", type=Path, default=Path("configs/toy_train_hard_level_b.yaml"))
    parser.add_argument("--hard-config", type=Path, default=Path("configs/toy_eval_hard.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints_toy_hard_b/best.pt"))
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--method", default="prism_full", choices=STAGE5_METHODS)
    parser.add_argument("--lambda-u-values", nargs="+", type=float, default=[0.0, 0.25, 0.5])
    parser.add_argument("--delta-values", nargs="+", type=float, default=[1.0, 1.2, 1.5])
    parser.add_argument("--goal-weight-values", nargs="+", type=float, default=[0.5, 0.8])
    parser.add_argument("--noise-values", nargs="+", type=float, default=[5.0, 8.0])
    parser.add_argument("--num-trajectories-values", nargs="+", type=int, default=[64])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/results/stage5_2_hard_planner_sweep.csv"),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Device to use for model inference.",
    )
    return parser.parse_args()


def _safe_mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _summarize_combo(
    *,
    method: str,
    config: dict[str, Any],
    hard: dict[str, Any],
    device: Any,
    model: Any,
    episodes: int,
    base_seed: int,
    combo: dict[str, Any],
) -> dict[str, Any]:
    run_config = copy.deepcopy(config)
    run_config["lambda_u"] = float(combo["lambda_u"])
    run_config["risk_threshold_delta"] = float(combo["risk_threshold_delta"])
    run_config["goal_weight"] = float(combo["goal_weight"])
    run_config["trajectory_noise_scale"] = float(combo["trajectory_noise_scale"])
    run_config["num_trajectories"] = int(combo["num_trajectories"])

    successes = 0
    collisions = 0
    timeouts = 0
    cumulative_risks: list[float] = []
    path_lengths: list[float] = []
    steps: list[float] = []
    feasible_ratios: list[float] = []
    selected_risks: list[float] = []
    selected_distances: list[float] = []

    for episode in range(episodes):
        seed = base_seed + episode
        rows, metrics = run_diagnostic_episode(
            method=method,
            config=run_config,
            hard=hard,
            device=device,
            seed=seed,
            episode=episode,
            model=model if method in MODEL_METHODS else None,
        )
        successes += int(bool(metrics["success"]))
        collisions += int(bool(metrics["collision"]))
        timeouts += int(bool(metrics["timeout"]))
        cumulative_risks.append(float(metrics["cumulative_risk"]))
        path_lengths.append(float(metrics["path_length"]))
        steps.append(float(metrics["steps"]))
        feasible_ratios.extend(float(row["feasible_ratio"]) for row in rows)
        selected_risks.extend(float(row["selected_traj_risk"]) for row in rows)
        selected_distances.extend(float(row["selected_final_distance"]) for row in rows)

    return {
        **combo,
        "episodes": episodes,
        "success_rate": successes / episodes,
        "collision_rate": collisions / episodes,
        "timeout_rate": timeouts / episodes,
        "avg_cumulative_risk": _safe_mean(cumulative_risks),
        "avg_path_length": _safe_mean(path_lengths),
        "avg_steps": _safe_mean(steps),
        "avg_feasible_ratio": _safe_mean(feasible_ratios),
        "avg_selected_traj_risk": _safe_mean(selected_risks),
        "avg_selected_final_distance": _safe_mean(selected_distances),
    }


def sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row["collision_rate"]),
        -float(row["success_rate"]),
        float(row["avg_cumulative_risk"]),
        float(row["avg_path_length"]),
    )


def print_top10(rows: list[dict[str, Any]]) -> None:
    print("Top 10 planner sweep configurations:")
    header = (
        "rank lambda_u delta goal_weight noise num_traj "
        "success collision avg_risk feasible selected_risk selected_final_dist"
    )
    print(header)
    for rank, row in enumerate(sorted(rows, key=sort_key)[:10], start=1):
        print(
            f"{rank:02d} "
            f"{float(row['lambda_u']):.2f} "
            f"{float(row['risk_threshold_delta']):.2f} "
            f"{float(row['goal_weight']):.2f} "
            f"{float(row['trajectory_noise_scale']):.2f} "
            f"{int(row['num_trajectories'])} "
            f"{float(row['success_rate']):.3f} "
            f"{float(row['collision_rate']):.3f} "
            f"{float(row['avg_cumulative_risk']):.3f} "
            f"{float(row['avg_feasible_ratio']):.3f} "
            f"{float(row['avg_selected_traj_risk']):.3f} "
            f"{float(row['avg_selected_final_distance']):.3f}"
        )


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError(f"--episodes must be positive, got {args.episodes}")
    config, hard = load_stage5_2_config(args.config, args.hard_config)
    base_seed = int(config.get("data", {}).get("seed", config.get("seed", 42)))
    device = resolve_device(args.device)
    model = build_loaded_model(config, args.checkpoint, device) if needs_model([args.method]) else None

    combos = [
        {
            "lambda_u": lambda_u,
            "risk_threshold_delta": delta,
            "goal_weight": goal_weight,
            "trajectory_noise_scale": noise,
            "num_trajectories": num_trajectories,
        }
        for lambda_u, delta, goal_weight, noise, num_trajectories in itertools.product(
            args.lambda_u_values,
            args.delta_values,
            args.goal_weight_values,
            args.noise_values,
            args.num_trajectories_values,
        )
    ]

    rows: list[dict[str, Any]] = []
    print(f"Stage-5.2 sweep method: {args.method}")
    print(f"Combinations: {len(combos)} episodes_per_combo={args.episodes}")
    for idx, combo in enumerate(combos, start=1):
        row = _summarize_combo(
            method=args.method,
            config=config,
            hard=hard,
            device=device,
            model=model,
            episodes=args.episodes,
            base_seed=base_seed,
            combo=combo,
        )
        rows.append(row)
        print(
            f"combo={idx}/{len(combos)} lambda_u={combo['lambda_u']} "
            f"delta={combo['risk_threshold_delta']} goal_weight={combo['goal_weight']} "
            f"noise={combo['trajectory_noise_scale']} num_traj={combo['num_trajectories']} "
            f"success={row['success_rate']:.3f} collision={row['collision_rate']:.3f} "
            f"feasible={row['avg_feasible_ratio']:.3f}"
        )

    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SWEEP_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print_top10(rows)
    print(f"Saved Stage-5.2 hard planner sweep to: {output}")


if __name__ == "__main__":
    main()
