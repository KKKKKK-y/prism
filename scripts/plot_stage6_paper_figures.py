from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RATE_METRICS = ["success_rate", "collision_rate", "timeout_rate"]
ALL_METRICS = [
    "success_rate",
    "collision_rate",
    "timeout_rate",
    "avg_cumulative_risk",
    "avg_path_length",
    "avg_steps",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Stage-6 paper-ready figures.")
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/results"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/visualizations"))
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv(path: Path) -> list[dict[str, str]]:
    path = resolve(path)
    if not path.exists():
        print(f"Warning: missing input, skipping plot: {path}")
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"Warning: empty input, skipping plot: {path}")
    return rows


def save(fig: plt.Figure, path: Path) -> None:
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved: {path}")


def method_labels(rows: list[dict[str, str]]) -> list[str]:
    return [row.get("display_name") or row.get("method", "") for row in rows]


def plot_main_comparison(results_dir: Path, output_dir: Path) -> None:
    rows = read_csv(results_dir / "stage6_main_comparison_table.csv")
    if not rows:
        return
    labels = method_labels(rows)
    colors = ["#64748b", "#2563eb", "#7c3aed", "#0891b2", "#16a34a"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for ax, metric in zip(axes.flatten(), ALL_METRICS):
        values = [float(row[metric]) for row in rows]
        ax.bar(labels, values, color=colors[: len(rows)])
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        if metric in RATE_METRICS:
            ax.set_ylim(0.0, 1.0)
    save(fig, output_dir / "stage6_main_comparison.png")


def plot_tradeoff(results_dir: Path, output_dir: Path) -> None:
    rows = read_csv(results_dir / "stage6_main_comparison_table.csv")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for row in rows:
        x = float(row["collision_rate"])
        y = float(row["success_rate"])
        label = row.get("display_name") or row["method"]
        color = "#16a34a" if row["method"] == "alpha_fusion_0.4" else "#2563eb"
        ax.scatter([x], [y], s=90, color=color)
        ax.annotate(label, (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("collision_rate")
    ax.set_ylabel("success_rate")
    ax.set_xlim(0.0, max(0.4, max(float(row["collision_rate"]) for row in rows) + 0.05))
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.set_title("Success-Collision Tradeoff")
    save(fig, output_dir / "stage6_success_collision_tradeoff.png")


def plot_alpha_sensitivity(results_dir: Path, output_dir: Path) -> None:
    rows = read_csv(results_dir / "stage6_alpha_sensitivity.csv")
    if not rows:
        return
    rows = sorted(rows, key=lambda row: float(row["alpha"]))
    alpha = [float(row["alpha"]) for row in rows]
    success = [float(row["success_rate"]) for row in rows]
    collision = [float(row["collision_rate"]) for row in rows]
    timeout = [float(row["timeout_rate"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(alpha, success, marker="o", label="success_rate", color="#16a34a")
    ax.plot(alpha, collision, marker="o", label="collision_rate", color="#dc2626")
    ax.plot(alpha, timeout, marker="o", label="timeout_rate", color="#64748b")
    ax.axvline(0.4, color="#111827", linestyle="--", linewidth=1.2, label="final alpha=0.4")
    ax.set_xlabel("alpha")
    ax.set_ylabel("rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_title("Alpha Sensitivity")
    save(fig, output_dir / "stage6_alpha_sensitivity.png")


def plot_ablation(results_dir: Path, output_dir: Path) -> None:
    rows = read_csv(results_dir / "stage6_ablation_table.csv")
    if not rows:
        return
    labels = method_labels(rows)
    x = list(range(len(rows)))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.bar([value - width / 2 for value in x], [float(row["success_rate"]) for row in rows], width, label="success_rate", color="#16a34a")
    ax.bar([value + width / 2 for value in x], [float(row["collision_rate"]) for row in rows], width, label="collision_rate", color="#dc2626")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    ax.set_title("Planner Ablation Comparison")
    save(fig, output_dir / "stage6_ablation_comparison.png")


def main() -> None:
    args = parse_args()
    plot_main_comparison(args.results_dir, args.output_dir)
    plot_tradeoff(args.results_dir, args.output_dir)
    plot_alpha_sensitivity(args.results_dir, args.output_dir)
    plot_ablation(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
