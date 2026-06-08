from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = [
    "success_rate",
    "collision_rate",
    "timeout_rate",
    "avg_cumulative_risk",
    "avg_path_length",
    "avg_steps",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PRISM Stage-5 baseline comparison metrics.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("outputs/results/stage5_baseline_results.csv"),
        help="Stage-5 baseline summary CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/visualizations/stage5_baseline_comparison.png"),
        help="Output PNG path.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Baseline results CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Baseline results CSV has no rows: {path}")
    return rows


def main() -> None:
    args = parse_args()
    rows = read_rows(args.csv.expanduser())
    methods = [row["method"] for row in rows]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes_flat = axes.flatten()
    colors = ["#2563eb", "#dc2626", "#7c3aed", "#0891b2", "#16a34a"]

    for ax, metric in zip(axes_flat, METRICS):
        values = [float(row[metric]) for row in rows]
        ax.bar(methods, values, color=colors[: len(methods)])
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.25)
        if metric.endswith("_rate"):
            ax.set_ylim(0.0, 1.0)

    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved baseline comparison plot to: {output_path}")


if __name__ == "__main__":
    main()
