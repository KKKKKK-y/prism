from __future__ import annotations

import argparse
from pathlib import Path

import torch

from prism.datasets import ToyFireRiskDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ToyFireRiskDataset npz file.")
    parser.add_argument("--npz", type=Path, default=Path("outputs/datasets/toy_fire_train.npz"), help="Path to toy npz file.")
    return parser.parse_args()


def stats(name: str, tensor: torch.Tensor) -> None:
    print(
        f"{name}.min={tensor.min().item():.6f}, "
        f"{name}.max={tensor.max().item():.6f}, "
        f"{name}.mean={tensor.mean().item():.6f}"
    )


def assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{name} contains NaN or Inf values.")


def main() -> None:
    args = parse_args()
    dataset = ToyFireRiskDataset(args.npz)
    obs = torch.from_numpy(dataset.obs)
    target = torch.from_numpy(dataset.target)

    print(f"obs.shape: {tuple(obs.shape)}")
    print(f"target.shape: {tuple(target.shape)}")
    stats("obs", obs)
    stats("target", target)

    assert tuple(obs.shape[1:]) == (4, 5, 64, 64), f"Unexpected obs shape suffix: {tuple(obs.shape[1:])}"
    assert tuple(target.shape[1:]) == (5, 1, 64, 64), f"Unexpected target shape suffix: {tuple(target.shape[1:])}"
    assert_finite("obs", obs)
    assert_finite("target", target)
    print("All PRISM Stage-4.2 toy dataset checks passed.")


if __name__ == "__main__":
    main()
