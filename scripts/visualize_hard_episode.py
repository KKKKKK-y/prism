from __future__ import annotations

import argparse
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

from prism.planners import STAGE5_METHODS
from prism.scripts.diagnose_hard_planner import (
    load_stage5_2_config,
    run_diagnostic_episode,
    validate_methods,
)
from prism.scripts.evaluate_baselines import MODEL_METHODS, build_loaded_model, needs_model
from prism.scripts.run_closed_loop import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize hard ToyFire trajectories for Stage-5.2 diagnostics.")
    parser.add_argument("--config", type=Path, default=Path("configs/toy_train_hard_level_b.yaml"))
    parser.add_argument("--hard-config", type=Path, default=Path("configs/toy_eval_hard.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints_toy_hard_b/best.pt"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods", nargs="+", default=STAGE5_METHODS)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/visualizations/stage5_2_hard_episode_seed0.png"),
    )
    parser.add_argument(
        "--failure-output",
        type=Path,
        default=None,
        help="Optional path for an automatically searched failure example plot.",
    )
    parser.add_argument("--find-failure", action="store_true", help="Search for a seed where --failure-method fails.")
    parser.add_argument("--failure-method", default="prism_full", help="Method used to search a failure example.")
    parser.add_argument("--search-limit", type=int, default=80, help="Maximum seeds to scan for a failure example.")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Device to use for model inference.",
    )
    return parser.parse_args()


def _run_methods_for_seed(
    *,
    seed: int,
    methods: list[str],
    config: dict[str, Any],
    hard: dict[str, Any],
    device: torch.device,
    model: torch.nn.Module | None,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for method in methods:
        _, metrics = run_diagnostic_episode(
            method=method,
            config=config,
            hard=hard,
            device=device,
            seed=seed,
            episode=0,
            model=model if method in MODEL_METHODS else None,
            record_trace=True,
        )
        results[method] = metrics
    return results


def _safe_risk_p95_map(metrics: dict[str, Any]) -> torch.Tensor | None:
    safe_risk = metrics.get("last_safe_risk_map")
    if not isinstance(safe_risk, torch.Tensor):
        return None
    if safe_risk.ndim == 4:
        safe_risk = safe_risk[:, 0]
    if safe_risk.ndim != 3:
        return None
    return torch.quantile(safe_risk.detach().cpu().float(), 0.95, dim=0)


def save_episode_plot(results: dict[str, dict[str, Any]], output: Path, title_seed: int) -> None:
    methods = list(results.keys())
    cols = 3
    rows = 2 if len(methods) > 3 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5.2 * rows), constrained_layout=True)
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, method in zip(axes_list, methods):
        metrics = results[method]
        risk_map = metrics["final_risk_map"].detach().cpu().float()
        fire_map = metrics["final_fire_map"].detach().cpu().float()
        smoke_map = metrics["final_smoke_map"].detach().cpu().float()
        obstacle_map = metrics["final_obstacle_map"].detach().cpu().float()
        path = metrics["path"].detach().cpu().float()
        start_xy = metrics["start_xy"].detach().cpu().float()
        goal_xy = metrics["goal_xy"].detach().cpu().float()

        ax.imshow(risk_map.numpy(), cmap="inferno", origin="upper", vmin=0.0, vmax=1.0)
        obstacle_mask = obstacle_map > 0.5
        if bool(obstacle_mask.any()):
            ax.imshow(obstacle_mask.numpy(), cmap="gray", origin="upper", alpha=0.38)
        ax.contour(fire_map.numpy(), levels=[0.35, 0.65], colors=["#fb923c", "#ef4444"], linewidths=0.8)
        ax.contour(smoke_map.numpy(), levels=[0.25, 0.50], colors=["#38bdf8", "#0ea5e9"], linewidths=0.55, alpha=0.7)

        if method == "prism_full":
            p95_map = _safe_risk_p95_map(metrics)
            if p95_map is not None:
                ax.contour(p95_map.numpy(), levels=[0.25, 0.50, 0.75], colors="#a7f3d0", linewidths=0.75, alpha=0.85)

        ax.plot(path[:, 0].numpy(), path[:, 1].numpy(), color="#e0f2fe", linewidth=2.0)
        ax.scatter([start_xy[0].item()], [start_xy[1].item()], marker="o", s=70, color="#22c55e", edgecolor="black")
        ax.scatter([goal_xy[0].item()], [goal_xy[1].item()], marker="*", s=120, color="#fde047", edgecolor="black")
        final_xy = path[-1]
        endpoint_color = "#22c55e" if metrics["success"] else "#ef4444"
        endpoint_marker = "o" if metrics["success"] else "x"
        ax.scatter([final_xy[0].item()], [final_xy[1].item()], marker=endpoint_marker, s=90, color=endpoint_color)

        extra = ""
        if method == "prism_full":
            p95_map = _safe_risk_p95_map(metrics)
            if p95_map is not None:
                extra = f"\nsafe_p95={float(p95_map.mean().item()):.3f}/{float(p95_map.max().item()):.3f}"
        ax.set_title(
            f"{method}\n"
            f"success={metrics['success']} collision={metrics['collision']} "
            f"risk={float(metrics['cumulative_risk']):.3f} steps={metrics['steps']}{extra}",
            fontsize=10,
        )
        ax.set_xlim(0, risk_map.shape[1] - 1)
        ax.set_ylim(risk_map.shape[0] - 1, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes_list[len(methods) :]:
        ax.axis("off")
    fig.suptitle(f"Stage-5.2 hard trajectory comparison, seed={title_seed}", fontsize=14)
    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def find_failure_seed(
    *,
    start_seed: int,
    search_limit: int,
    failure_method: str,
    methods: list[str],
    config: dict[str, Any],
    hard: dict[str, Any],
    device: torch.device,
    model: torch.nn.Module | None,
) -> tuple[int, dict[str, dict[str, Any]]]:
    for seed in range(start_seed, start_seed + search_limit):
        results = _run_methods_for_seed(
            seed=seed,
            methods=methods,
            config=config,
            hard=hard,
            device=device,
            model=model,
        )
        target = results[failure_method]
        if bool(target["collision"]) or bool(target["timeout"]):
            return seed, results
    fallback = _run_methods_for_seed(
        seed=start_seed,
        methods=methods,
        config=config,
        hard=hard,
        device=device,
        model=model,
    )
    return start_seed, fallback


def main() -> None:
    args = parse_args()
    methods = validate_methods(args.methods)
    if args.failure_method not in methods:
        raise ValueError(f"--failure-method must be one of selected methods, got {args.failure_method!r}")
    config, hard = load_stage5_2_config(args.config, args.hard_config)
    device = resolve_device(args.device)
    model = build_loaded_model(config, args.checkpoint, device) if needs_model(methods) else None

    results = _run_methods_for_seed(
        seed=args.seed,
        methods=methods,
        config=config,
        hard=hard,
        device=device,
        model=model,
    )
    save_episode_plot(results, args.output, title_seed=args.seed)
    print(f"Saved hard episode visualization to: {args.output}")

    if args.find_failure:
        failure_output = args.failure_output or Path("outputs/visualizations/stage5_2_hard_episode_failure_examples.png")
        failure_seed, failure_results = find_failure_seed(
            start_seed=args.seed,
            search_limit=args.search_limit,
            failure_method=args.failure_method,
            methods=methods,
            config=config,
            hard=hard,
            device=device,
            model=model,
        )
        save_episode_plot(failure_results, failure_output, title_seed=failure_seed)
        print(
            f"Saved hard failure visualization to: {failure_output} "
            f"(seed={failure_seed}, method={args.failure_method})"
        )


if __name__ == "__main__":
    main()
