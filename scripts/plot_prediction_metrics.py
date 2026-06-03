from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PRISM Stage-4.3 toy prediction metrics by horizon.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("outputs/results/stage4_toy_prediction_metrics.csv"),
        help="Prediction metrics CSV written by scripts/evaluate_prediction_on_toy.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/visualizations/stage4_toy_prediction_metrics.png"),
        help="Output PNG path.",
    )
    return parser.parse_args()


def read_metric_rows(csv_path: Path) -> list[dict[str, float]]:
    csv_path = csv_path.expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"Prediction metrics CSV not found: {csv_path}")

    rows: list[dict[str, float]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "horizon": float(row["horizon"]),
                    "mae": float(row["mae"]),
                    "rmse": float(row["rmse"]),
                }
            )
    if not rows:
        raise ValueError(f"Prediction metrics CSV is empty: {csv_path}")
    return rows


def main() -> None:
    args = parse_args()
    rows = read_metric_rows(args.csv)

    horizons = [int(row["horizon"]) for row in rows]
    labels = [f"t+{h}" for h in horizons]

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.plot(labels, [row["mae"] for row in rows], label="MAE", marker="o")
    ax.plot(labels, [row["rmse"] for row in rows], label="RMSE", marker="o")
    ax.set_title("Toy Prediction Error by Horizon")
    ax.set_xlabel("Prediction Horizon")
    ax.set_ylabel("Error")
    ax.grid(True, alpha=0.3)
    ax.legend()

    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved prediction metrics plot to: {output_path}")


if __name__ == "__main__":
    main()
