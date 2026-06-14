from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from prism.datasets import ToyFireRiskDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate hard ToyFire npz dataset and metadata.")
    parser.add_argument(
        "--npz",
        type=Path,
        default=Path("outputs/datasets_hard/toy_fire_train_hard.npz"),
        help="Path to hard toy npz file.",
    )
    return parser.parse_args()


def stats(name: str, tensor: torch.Tensor) -> None:
    print(
        f"{name}.min={tensor.min().item():.6f}, "
        f"{name}.max={tensor.max().item():.6f}, "
        f"{name}.mean={tensor.mean().item():.6f}"
    )


def main() -> None:
    args = parse_args()
    dataset = ToyFireRiskDataset(args.npz)
    obs = torch.from_numpy(dataset.obs)
    target = torch.from_numpy(dataset.target)

    print(f"obs.shape: {tuple(obs.shape)}")
    print(f"target.shape: {tuple(target.shape)}")
    stats("obs", obs)
    stats("target", target)

    if tuple(obs.shape[1:]) != (4, 5, 64, 64):
        raise AssertionError(f"Unexpected obs shape suffix: {tuple(obs.shape[1:])}")
    if tuple(target.shape[1:]) != (5, 1, 64, 64):
        raise AssertionError(f"Unexpected target shape suffix: {tuple(target.shape[1:])}")
    if not torch.isfinite(obs).all():
        raise AssertionError("obs contains NaN or Inf values.")
    if not torch.isfinite(target).all():
        raise AssertionError("target contains NaN or Inf values.")

    with np.load(args.npz) as data:
        if "metadata" not in data:
            raise AssertionError(f"{args.npz} does not contain hard metadata.")
        metadata = json.loads(str(data["metadata"].item()))
    if metadata.get("mode") != "hard":
        raise AssertionError(f"Expected metadata mode='hard', got {metadata.get('mode')!r}")
    print(f"metadata.mode: {metadata.get('mode')}")
    print("All PRISM hard toy dataset checks passed.")


if __name__ == "__main__":
    main()
