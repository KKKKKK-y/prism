from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prism.config import load_config
from prism.datasets import DummyFireRiskDataset
from prism.models import RiskPredictor
from prism.trainers.trainer import select_device
from prism.utils.uncertainty import compute_safe_risk, mc_dropout_predict, propagate_uncertainty


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize PRISM Stage-2 uncertainty propagation.")
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"), help="Path to YAML config.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("prism/outputs/checkpoints/best.pt"),
        help="Path to Stage-1 checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("prism/outputs/visualizations/stage2_uncertainty.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Device to use for inference.",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return select_device()
    if name == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def build_dummy_dataset(config: dict[str, Any]) -> DummyFireRiskDataset:
    return DummyFireRiskDataset(
        num_samples=1,
        obs_window=int(config.get("obs_window", 4)),
        horizon=int(config.get("horizon", 5)),
        channels=int(config.get("channels", 6)),
        image_size=int(config.get("image_size", 64)),
        seed=int(config.get("data", {}).get("seed", 42)),
    )


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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    num_mc_samples = int(config.get("num_mc_samples", 5))
    alpha = float(config.get("uncertainty_alpha", 0.7))
    lambda_u = float(config.get("lambda_u", 0.5))
    horizon = int(config.get("horizon", 5))

    dataset = build_dummy_dataset(config)
    sample = dataset[0]
    # obs: [1, k, C, 64, 64]
    obs = sample["obs"].unsqueeze(0)

    device = resolve_device(args.device)
    model = build_model(config).to(device)
    load_checkpoint_if_available(model, args.checkpoint, device)

    with torch.no_grad():
        # mu_mean/sigma_mc: [1, H, 1, 64, 64], mu_samples: [N, 1, H, 1, 64, 64]
        mu_mean, sigma_mc, mu_samples = mc_dropout_predict(model, obs.to(device), num_samples=num_mc_samples)
        # sigma_prop/propagated_var: [1, H, 1, 64, 64]
        sigma_prop, _ = propagate_uncertainty(mu_samples, alpha=alpha)
        # safe_risk: [1, H, 1, 64, 64]
        safe_risk = compute_safe_risk(mu_mean, sigma_prop, lambda_u=lambda_u)

    rows = [
        ("Predicted Mean Risk μ", mu_mean),
        ("MC Uncertainty σ_mc", sigma_mc),
        ("Propagated Uncertainty σ_prop", sigma_prop),
        ("Safe Risk μ+λσ", safe_risk),
    ]

    fig, axes = plt.subplots(4, horizon, figsize=(3.4 * horizon, 12), constrained_layout=True)
    for row_idx, (row_title, tensor) in enumerate(rows):
        for h in range(horizon):
            ax = axes[row_idx, h]
            image = to_image(tensor[0, h, 0])
            im = ax.imshow(image.numpy(), cmap="inferno")
            ax.set_title(f"{row_title}\nt+{h + 1}", fontsize=10)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved uncertainty visualization to: {output_path}")


if __name__ == "__main__":
    main()
