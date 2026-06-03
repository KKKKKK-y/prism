from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PRISM Stage-4.3 toy training curves from CSV logs.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("outputs/results/stage4_toy_training_log.csv"),
        help="Training log CSV written by scripts/train.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/visualizations/stage4_toy_training_curve.png"),
        help="Output PNG path.",
    )
    return parser.parse_args()


def read_training_rows(csv_path: Path) -> list[dict[str, float]]:
    csv_path = csv_path.expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"Training log CSV not found: {csv_path}")

    rows: list[dict[str, float]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {"epoch": float(row["epoch"])}
            for key in ("train_loss", "train_pred_loss", "train_unc_loss", "val_loss", "val_pred_loss", "val_unc_loss"):
                value = row.get(key, "")
                parsed[key] = float(value) if value not in {"", None} else float("nan")
            rows.append(parsed)
    if not rows:
        raise ValueError(f"Training log CSV is empty: {csv_path}")
    return rows


def main() -> None:
    args = parse_args()
    rows = read_training_rows(args.csv)

    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    axes[0].plot(epochs, [row["train_loss"] for row in rows], label="Train Total", marker="o")
    axes[0].plot(epochs, [row["val_loss"] for row in rows], label="Val Total", marker="o")
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, [row["train_pred_loss"] for row in rows], label="Train Prediction", marker="o")
    axes[1].plot(epochs, [row["val_pred_loss"] for row in rows], label="Val Prediction", marker="o")
    axes[1].plot(epochs, [row["train_unc_loss"] for row in rows], label="Train Uncertainty", linestyle="--")
    axes[1].plot(epochs, [row["val_unc_loss"] for row in rows], label="Val Uncertainty", linestyle="--")
    axes[1].set_title("Loss Components")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved training curve to: {output_path}")


if __name__ == "__main__":
    main()
