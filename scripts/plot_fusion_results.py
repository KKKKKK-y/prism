from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Stage-5.8 fusion sweep results.")
    parser.add_argument("--input", type=Path, default=Path("outputs/results/stage5_8_fusion_all_checkpoints.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/visualizations/stage5_8_fusion_comparison.png"))
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def label(row: dict[str, str]) -> str:
    return f"{row['checkpoint_name']}\n{row['mode']}\na={row['alpha']} s={row['scale']}"


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    top = rows[:10]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)

    labels = [label(row) for row in top]
    success = [float(row["success_rate"]) for row in top]
    collision = [float(row["collision_rate"]) for row in top]
    avg_risk = [float(row["avg_cumulative_risk"]) for row in top]

    axes[0, 0].bar(labels, success, color="#2563eb")
    axes[0, 0].set_title("Top 10 Success Rate")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].tick_params(axis="x", rotation=45)

    axes[0, 1].bar(labels, collision, color="#dc2626")
    axes[0, 1].set_title("Top 10 Collision Rate")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].tick_params(axis="x", rotation=45)

    all_success = [float(row["success_rate"]) for row in rows]
    all_collision = [float(row["collision_rate"]) for row in rows]
    all_risk = [float(row["avg_cumulative_risk"]) for row in rows]
    axes[1, 0].scatter(all_collision, all_success, c="#059669", alpha=0.75)
    axes[1, 0].set_title("Success vs Collision")
    axes[1, 0].set_xlabel("collision_rate")
    axes[1, 0].set_ylabel("success_rate")
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].scatter(all_risk, all_success, c="#7c3aed", alpha=0.75)
    axes[1, 1].set_title("Average Risk vs Success")
    axes[1, 1].set_xlabel("avg_cumulative_risk")
    axes[1, 1].set_ylabel("success_rate")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(alpha=0.25)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"Saved Stage-5.8 fusion plot to: {args.output}")


if __name__ == "__main__":
    main()
