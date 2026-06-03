from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from prism.config import load_config
from prism.datasets import DummyFireRiskDataset
from prism.models import RiskPredictor
from prism.trainers.trainer import select_device
from prism.utils.uncertainty import compute_safe_risk, mc_dropout_predict, propagate_uncertainty


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate PRISM Stage-2 uncertainty propagation.")
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"), help="Path to YAML config.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("prism/outputs/checkpoints/best.pt"),
        help="Path to Stage-1 checkpoint.",
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


def build_dummy_loader(config: dict[str, Any]) -> DataLoader:
    dataset = DummyFireRiskDataset(
        num_samples=max(int(config.get("batch_size", 2)), 2),
        obs_window=int(config.get("obs_window", 4)),
        horizon=int(config.get("horizon", 5)),
        channels=int(config.get("channels", 6)),
        image_size=int(config.get("image_size", 64)),
        seed=int(config.get("data", {}).get("seed", 42)),
    )
    return DataLoader(dataset, batch_size=int(config.get("batch_size", 2)), shuffle=False, num_workers=0)


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


def assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{name} contains NaN or Inf values.")


def print_stats(name: str, tensor: torch.Tensor) -> None:
    values = tensor.detach().cpu()
    print(
        f"{name}.min={values.min().item():.6f}, "
        f"{name}.max={values.max().item():.6f}, "
        f"{name}.mean={values.mean().item():.6f}"
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    batch_size = int(config.get("batch_size", 2))
    horizon = int(config.get("horizon", 5))
    image_size = int(config.get("image_size", 64))
    num_mc_samples = int(config.get("num_mc_samples", 5))
    alpha = float(config.get("uncertainty_alpha", 0.7))
    lambda_u = float(config.get("lambda_u", 0.5))

    loader = build_dummy_loader(config)
    batch = next(iter(loader))
    obs = batch["obs"]
    target = batch["target"]

    device = resolve_device(args.device)
    model = build_model(config).to(device)
    load_checkpoint_if_available(model, args.checkpoint, device)

    # obs: [B, k, C, 64, 64]
    obs = obs.to(device)
    # target: [B, H, 1, 64, 64]
    target = target.to(device)
    # mu_mean/sigma_mc: [B, H, 1, 64, 64], mu_samples: [N, B, H, 1, 64, 64]
    mu_mean, sigma_mc, mu_samples = mc_dropout_predict(model, obs, num_samples=num_mc_samples)
    # sigma_prop/propagated_var: [B, H, 1, 64, 64]
    sigma_prop, _ = propagate_uncertainty(mu_samples, alpha=alpha)
    # safe_risk: [B, H, 1, 64, 64]
    safe_risk = compute_safe_risk(mu_mean, sigma_prop, lambda_u=lambda_u)

    expected_samples = (num_mc_samples, batch_size, horizon, 1, image_size, image_size)
    expected_risk = (batch_size, horizon, 1, image_size, image_size)

    print(f"obs.shape: {tuple(obs.shape)}")
    print(f"target.shape: {tuple(target.shape)}")
    print(f"mu_samples.shape: {tuple(mu_samples.shape)}")
    print(f"mu_mean.shape: {tuple(mu_mean.shape)}")
    print(f"sigma_mc.shape: {tuple(sigma_mc.shape)}")
    print(f"sigma_prop.shape: {tuple(sigma_prop.shape)}")
    print(f"safe_risk.shape: {tuple(safe_risk.shape)}")

    assert tuple(mu_samples.shape) == expected_samples, (
        f"mu_samples.shape expected {expected_samples}, got {tuple(mu_samples.shape)}"
    )
    assert tuple(mu_mean.shape) == expected_risk, f"mu_mean.shape expected {expected_risk}, got {tuple(mu_mean.shape)}"
    assert tuple(sigma_mc.shape) == expected_risk, f"sigma_mc.shape expected {expected_risk}, got {tuple(sigma_mc.shape)}"
    assert tuple(sigma_prop.shape) == expected_risk, (
        f"sigma_prop.shape expected {expected_risk}, got {tuple(sigma_prop.shape)}"
    )
    assert tuple(safe_risk.shape) == expected_risk, (
        f"safe_risk.shape expected {expected_risk}, got {tuple(safe_risk.shape)}"
    )

    assert_finite("mu_samples", mu_samples)
    assert_finite("mu_mean", mu_mean)
    assert_finite("sigma_mc", sigma_mc)
    assert_finite("sigma_prop", sigma_prop)
    assert_finite("safe_risk", safe_risk)

    print_stats("mu_mean", mu_mean)
    print_stats("sigma_mc", sigma_mc)
    print_stats("sigma_prop", sigma_prop)
    print_stats("safe_risk", safe_risk)
    print("All PRISM Stage-2 uncertainty propagation checks passed.")


if __name__ == "__main__":
    main()
