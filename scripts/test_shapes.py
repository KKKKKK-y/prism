from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from prism.config import load_config
from prism.datasets import DummyFireRiskDataset
from prism.models import RiskPredictor
from prism.trainers.trainer import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate PRISM Stage-1 tensor shapes and numeric ranges.")
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"), help="Path to YAML config.")
    return parser.parse_args()


def build_dummy_loader(config: dict) -> DataLoader:
    dataset = DummyFireRiskDataset(
        num_samples=max(int(config.get("batch_size", 2)), 2),
        obs_window=int(config.get("obs_window", 4)),
        horizon=int(config.get("horizon", 5)),
        channels=int(config.get("channels", 6)),
        image_size=int(config.get("image_size", 64)),
        seed=int(config.get("data", {}).get("seed", 42)),
    )
    return DataLoader(dataset, batch_size=int(config.get("batch_size", 2)), shuffle=False, num_workers=0)


def build_model(config: dict) -> RiskPredictor:
    return RiskPredictor(
        in_channels=int(config.get("channels", 6)),
        obs_window=int(config.get("obs_window", 4)),
        horizon=int(config.get("horizon", 5)),
        base_channels=int(config.get("base_channels", 32)),
        latent_channels=int(config.get("latent_channels", 128)),
        gru_layers=int(config.get("gru_layers", 1)),
        dropout_p=float(config.get("dropout_p", 0.1)),
    )


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
    obs_window = int(config.get("obs_window", 4))
    channels = int(config.get("channels", 6))
    horizon = int(config.get("horizon", 5))
    image_size = int(config.get("image_size", 64))
    lambda_u = float(config.get("lambda_u", 1.0))

    loader = build_dummy_loader(config)
    batch = next(iter(loader))
    obs = batch["obs"]
    target = batch["target"]

    device = select_device()
    model = build_model(config).to(device)
    model.eval()

    with torch.no_grad():
        mu, log_var = model(obs.to(device))
        sigma = torch.sqrt(torch.exp(log_var))
        safe_risk = mu + lambda_u * sigma

    expected_obs = (batch_size, obs_window, channels, image_size, image_size)
    expected_risk = (batch_size, horizon, 1, image_size, image_size)

    print(f"obs.shape: {tuple(obs.shape)}")
    print(f"target.shape: {tuple(target.shape)}")
    print(f"mu.shape: {tuple(mu.shape)}")
    print(f"log_var.shape: {tuple(log_var.shape)}")
    print(f"sigma.shape: {tuple(sigma.shape)}")
    print(f"safe_risk.shape: {tuple(safe_risk.shape)}")

    assert tuple(obs.shape) == expected_obs, f"obs.shape expected {expected_obs}, got {tuple(obs.shape)}"
    assert tuple(target.shape) == expected_risk, f"target.shape expected {expected_risk}, got {tuple(target.shape)}"
    assert tuple(mu.shape) == expected_risk, f"mu.shape expected {expected_risk}, got {tuple(mu.shape)}"
    assert tuple(log_var.shape) == expected_risk, f"log_var.shape expected {expected_risk}, got {tuple(log_var.shape)}"
    assert tuple(sigma.shape) == expected_risk, f"sigma.shape expected {expected_risk}, got {tuple(sigma.shape)}"
    assert tuple(safe_risk.shape) == expected_risk, f"safe_risk.shape expected {expected_risk}, got {tuple(safe_risk.shape)}"

    assert_finite("mu", mu)
    assert_finite("log_var", log_var)
    assert_finite("sigma", sigma)
    assert_finite("safe_risk", safe_risk)

    print_stats("mu", mu)
    print_stats("sigma", sigma)
    print_stats("safe_risk", safe_risk)
    print("All PRISM Stage-1 shape and numeric checks passed.")


if __name__ == "__main__":
    main()
