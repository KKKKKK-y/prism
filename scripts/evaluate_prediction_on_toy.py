from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from prism.config import load_config
from prism.datasets import ToyFireRiskDataset
from prism.models import RiskPredictor
from prism.trainers.trainer import select_device

try:
    from skimage.metrics import structural_similarity as ssim
except Exception:  # pragma: no cover - optional dependency
    ssim = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PRISM prediction quality on ToyFireEnv test npz.")
    parser.add_argument("--config", type=Path, default=Path("configs/toy_train.yaml"), help="Path to YAML config.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/checkpoints_toy/best.pt"),
        help="Path to trained toy checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/results/stage4_toy_prediction_metrics.csv"),
        help="Output metrics CSV path.",
    )
    return parser.parse_args()


def build_model(config: dict) -> RiskPredictor:
    return RiskPredictor(
        in_channels=int(config.get("channels", 5)),
        obs_window=int(config.get("obs_window", 4)),
        horizon=int(config.get("horizon", 5)),
        base_channels=int(config.get("base_channels", 16)),
        latent_channels=int(config.get("latent_channels", 64)),
        gru_layers=int(config.get("gru_layers", 1)),
        dropout_p=float(config.get("dropout_p", 0.1)),
    )


def load_checkpoint_if_available(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint_path = checkpoint_path.expanduser()
    if not checkpoint_path.exists():
        print(f"Warning: checkpoint not found at {checkpoint_path}. Using randomly initialized model.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint from: {checkpoint_path}")


def compute_ssim_for_horizon(pred: torch.Tensor, target: torch.Tensor) -> list[float]:
    if ssim is None:
        return []
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    values = []
    for idx in range(pred_np.shape[0]):
        values.append(ssim(target_np[idx], pred_np[idx], data_range=1.0))
    return [float(value) for value in values]


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = select_device()

    dataset = ToyFireRiskDataset(config.get("test_npz", "outputs/datasets/toy_fire_test.npz"))
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 16)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
    )
    model = build_model(config).to(device)
    load_checkpoint_if_available(model, args.checkpoint, device)
    model.eval()

    horizon = int(config.get("horizon", 5))
    abs_sum = torch.zeros(horizon, dtype=torch.float64)
    sq_sum = torch.zeros(horizon, dtype=torch.float64)
    count = 0
    ssim_values: list[list[float]] = [[] for _ in range(horizon)]

    for batch in loader:
        # obs: [B, k, C, 64, 64], target: [B, H, 1, 64, 64]
        obs = batch["obs"].to(device)
        target = batch["target"].to(device)
        mu, _ = model(obs)
        diff = mu - target
        abs_sum += diff.abs().sum(dim=(0, 2, 3, 4)).detach().cpu().double()
        sq_sum += diff.pow(2).sum(dim=(0, 2, 3, 4)).detach().cpu().double()
        count += target.shape[0] * target.shape[-1] * target.shape[-2]

        if ssim is not None:
            for h in range(horizon):
                ssim_values[h].extend(compute_ssim_for_horizon(mu[:, h, 0], target[:, h, 0]))

    rows = []
    if ssim is None:
        print("Warning: scikit-image is not installed. Skipping SSIM.")

    for h in range(horizon):
        mae = (abs_sum[h] / count).item()
        rmse = torch.sqrt(sq_sum[h] / count).item()
        ssim_mean = "" if not ssim_values[h] else sum(ssim_values[h]) / len(ssim_values[h])
        row = {
            "horizon": h + 1,
            "mae": mae,
            "rmse": rmse,
            "ssim": ssim_mean,
        }
        rows.append(row)
        if ssim_mean == "":
            print(f"Horizon t+{h + 1}: MAE={mae:.6f}, RMSE={rmse:.6f}")
        else:
            print(f"Horizon t+{h + 1}: MAE={mae:.6f}, RMSE={rmse:.6f}, SSIM={ssim_mean:.6f}")

    overall_count = count * horizon
    overall_mae = (abs_sum.sum() / overall_count).item()
    overall_rmse = torch.sqrt(sq_sum.sum() / overall_count).item()
    print(f"Overall: MAE={overall_mae:.6f}, RMSE={overall_rmse:.6f}")

    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["horizon", "mae", "rmse", "ssim"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved toy prediction metrics to: {output_path}")


if __name__ == "__main__":
    main()
