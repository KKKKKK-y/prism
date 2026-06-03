from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prism.config import load_config
from prism.datasets import DummyFireRiskDataset, ToyFireRiskDataset
from prism.models import RiskPredictor
from prism.trainers.trainer import select_device
from prism.utils.uncertainty import compute_safe_risk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize one PRISM Stage-1 prediction.")
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"), help="Path to YAML config.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("prism/outputs/checkpoints/best.pt"),
        help="Path to a PRISM checkpoint. Missing checkpoints fall back to random weights.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to <config output_dir>/visualizations/stage1_prediction.png.",
    )
    parser.add_argument(
        "--all_horizons",
        action="store_true",
        help="Save a 3xH target/prediction/error grid for every prediction horizon.",
    )
    return parser.parse_args()


def build_dummy_dataset(config: dict[str, Any]) -> DummyFireRiskDataset:
    return DummyFireRiskDataset(
        num_samples=1,
        obs_window=int(config.get("obs_window", 4)),
        horizon=int(config.get("horizon", 5)),
        channels=int(config.get("channels", 6)),
        image_size=int(config.get("image_size", 64)),
        seed=int(config.get("data", {}).get("seed", 42)),
    )


def load_sample(config: dict[str, Any]) -> dict[str, torch.Tensor]:
    if config.get("dataset_type") == "toy_npz":
        dataset = ToyFireRiskDataset(config.get("val_npz", "outputs/datasets/toy_fire_val.npz"))
        return dataset[0]

    dataset = build_dummy_dataset(config)
    return dataset[0]


def build_model(config: dict[str, Any]) -> RiskPredictor:
    return RiskPredictor(
        in_channels=int(config.get("channels", 6)),
        obs_window=int(config.get("obs_window", 4)),
        horizon=int(config.get("horizon", 5)),
        base_channels=int(config.get("base_channels", 32)),
        latent_channels=int(config.get("latent_channels", 128)),
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


def to_image(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float().cpu().squeeze()


def resolve_output_path(config: dict[str, Any], output: Path | None, all_horizons: bool = False) -> Path:
    if output is not None:
        return output.expanduser()
    if all_horizons:
        visualization_dir = Path(config.get("visualization_dir", Path(config.get("output_dir", "prism/outputs")) / "visualizations"))
        filename = "stage4_toy_prediction_horizons.png" if config.get("dataset_type") == "toy_npz" else "stage1_prediction_horizons.png"
        return visualization_dir / filename
    return Path(config.get("output_dir", "prism/outputs")) / "visualizations" / "stage1_prediction.png"


def plot_all_horizons(
    target: torch.Tensor,
    mu: torch.Tensor,
    output_path: Path,
) -> None:
    # target/mu: [1, H, 1, 64, 64]. Rows are target, predicted mean and absolute error.
    horizon = target.shape[1]
    error = (mu - target).abs()
    fig, axes = plt.subplots(3, horizon, figsize=(3.2 * horizon, 9), constrained_layout=True)
    row_titles = ["Target Risk", "Predicted Mean Risk μ", "Absolute Error |μ-target|"]
    tensors = [target, mu, error]

    for row_idx, (row_title, tensor) in enumerate(zip(row_titles, tensors)):
        for h in range(horizon):
            ax = axes[row_idx, h] if horizon > 1 else axes[row_idx]
            image = to_image(tensor[0, h, 0])
            im = ax.imshow(image.numpy(), cmap="inferno", vmin=0.0, vmax=1.0)
            ax.set_title(f"{row_title} t+{h + 1}", fontsize=10)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    lambda_u = float(config.get("lambda_u", 1.0))

    sample = load_sample(config)
    obs = sample["obs"].unsqueeze(0)
    target = sample["target"].unsqueeze(0)

    device = select_device()
    model = build_model(config).to(device)
    load_checkpoint_if_available(model, args.checkpoint, device)
    model.eval()

    with torch.no_grad():
        mu, log_var = model(obs.to(device))
        sigma = torch.sqrt(torch.exp(log_var))
        safe_risk = compute_safe_risk(mu, sigma, lambda_u=lambda_u)

    output_path = resolve_output_path(config, args.output, all_horizons=args.all_horizons)
    if args.all_horizons:
        plot_all_horizons(target=target, mu=mu.cpu(), output_path=output_path)
        print(f"Saved visualization to: {output_path}")
        return

    risk_channel = 3 if config.get("dataset_type") == "toy_npz" else (5 if obs.shape[2] >= 6 else obs.shape[2] - 1)
    panels = [
        ("Current Risk Map", to_image(obs[0, -1, risk_channel])),
        ("Target Risk t+1", to_image(target[0, 0, 0])),
        ("Predicted Mean Risk μ t+1", to_image(mu[0, 0, 0])),
        ("Predictive Uncertainty σ t+1", to_image(sigma[0, 0, 0])),
        ("Safe Risk μ+λσ t+1", to_image(safe_risk[0, 0, 0])),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4), constrained_layout=True)
    for ax, (title, image) in zip(axes, panels):
        im = ax.imshow(image.numpy(), cmap="inferno")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved visualization to: {output_path}")


if __name__ == "__main__":
    main()
