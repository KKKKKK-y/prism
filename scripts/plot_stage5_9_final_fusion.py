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
    parser = argparse.ArgumentParser(description="Plot Stage-5.9 final PRISM fusion evaluation.")
    parser.add_argument("--input", type=Path, default=Path("outputs/results/stage5_9_final_fusion_results.csv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/visualizations/stage5_9_final_fusion_comparison.png"),
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    methods = [row["method"] for row in rows]
    colors = ["#64748b", "#2563eb", "#7c3aed", "#0891b2", "#16a34a"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, metric in zip(axes.flatten(), METRICS):
        values = [float(row[metric]) for row in rows]
        ax.bar(methods, values, color=colors[: len(methods)])
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        if metric.endswith("_rate"):
            ax.set_ylim(0.0, 1.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"Saved Stage-5.9 final fusion plot to: {args.output}")


if __name__ == "__main__":
    main()
