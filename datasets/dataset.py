from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class SyntheticFireRiskDataset(Dataset):
    """Small procedural dataset for smoke-testing the training pipeline."""

    def __init__(
        self,
        num_samples: int = 128,
        obs_window: int = 4,
        horizon: int = 5,
        channels: int = 6,
        image_size: int = 64,
        seed: int = 42,
    ) -> None:
        self.num_samples = num_samples
        self.obs_window = obs_window
        self.horizon = horizon
        self.channels = channels
        self.image_size = image_size
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(self.seed + idx)
        size = self.image_size

        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, size),
            torch.linspace(-1.0, 1.0, size),
            indexing="ij",
        )
        cx = torch.empty(1).uniform_(-0.45, 0.45, generator=generator).item()
        cy = torch.empty(1).uniform_(-0.45, 0.45, generator=generator).item()
        vx = torch.empty(1).uniform_(-0.08, 0.08, generator=generator).item()
        vy = torch.empty(1).uniform_(-0.08, 0.08, generator=generator).item()
        sigma = torch.empty(1).uniform_(0.18, 0.34, generator=generator).item()

        frames = []
        total_steps = self.obs_window + self.horizon
        for t in range(total_steps):
            center_x = cx + vx * t
            center_y = cy + vy * t
            dist2 = (xx - center_x).pow(2) + (yy - center_y).pow(2)
            risk = torch.exp(-dist2 / (2.0 * sigma**2)).clamp(0.0, 1.0)
            thermal = (risk + 0.05 * torch.randn(size, size, generator=generator)).clamp(0.0, 1.0)
            occupancy = ((xx.abs() > 0.82) | (yy.abs() > 0.82)).float()

            rgb = torch.stack(
                [
                    (0.25 + 0.70 * risk).clamp(0.0, 1.0),
                    (0.20 + 0.20 * torch.randn(size, size, generator=generator)).clamp(0.0, 1.0),
                    (0.15 + 0.15 * (1.0 - risk)).clamp(0.0, 1.0),
                ],
                dim=0,
            )
            base = torch.cat([rgb, thermal.unsqueeze(0), occupancy.unsqueeze(0), risk.unsqueeze(0)], dim=0)
            if self.channels > 6:
                extras = torch.zeros(self.channels - 6, size, size)
                base = torch.cat([base, extras], dim=0)
            frames.append(base[: self.channels])

        sequence = torch.stack(frames, dim=0).float()
        risk_channel = 5 if self.channels >= 6 else self.channels - 1
        target = sequence[self.obs_window :, risk_channel : risk_channel + 1, :, :].contiguous()
        obs = sequence[: self.obs_window].contiguous()
        return {"obs": obs, "target": target}


class FireRiskDataset(Dataset):
    """
    Dataset for PRISM fire risk forecasting.

    Supported sample formats:
    - .pt/.pth file containing {"obs": Tensor, "target": Tensor}
    - .npz file containing obs and target arrays

    obs:    [obs_window, C, 64, 64]
    target: [horizon, 1, 64, 64]
    """

    def __init__(
        self,
        root: str | Path,
        obs_window: int = 4,
        horizon: int = 5,
        image_size: int = 64,
    ) -> None:
        self.root = Path(root).expanduser()
        self.obs_window = obs_window
        self.horizon = horizon
        self.image_size = image_size

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        patterns = ("*.pt", "*.pth", "*.npz")
        self.files = sorted(path for pattern in patterns for path in self.root.rglob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No .pt/.pth/.npz samples found under: {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path = self.files[idx]
        sample = self._load_sample(path)
        obs = torch.as_tensor(sample["obs"], dtype=torch.float32)
        target = torch.as_tensor(sample["target"], dtype=torch.float32)
        self._validate_shapes(obs, target, path)
        return {"obs": obs, "target": target}

    def _load_sample(self, path: Path) -> dict[str, Any]:
        if path.suffix in {".pt", ".pth"}:
            sample = torch.load(path, map_location="cpu")
            if not isinstance(sample, dict):
                raise TypeError(f"Expected dict sample in {path}, got {type(sample)!r}")
            return sample

        if path.suffix == ".npz":
            with np.load(path) as data:
                return {"obs": data["obs"], "target": data["target"]}

        raise ValueError(f"Unsupported sample extension: {path.suffix}")

    def _validate_shapes(self, obs: torch.Tensor, target: torch.Tensor, path: Path) -> None:
        if obs.ndim != 4:
            raise ValueError(f"{path}: obs must be [T,C,H,W], got {tuple(obs.shape)}")
        if target.ndim != 4:
            raise ValueError(f"{path}: target must be [H,1,H,W], got {tuple(target.shape)}")
        if obs.shape[0] != self.obs_window:
            raise ValueError(f"{path}: expected obs_window={self.obs_window}, got {obs.shape[0]}")
        if target.shape[0] != self.horizon:
            raise ValueError(f"{path}: expected horizon={self.horizon}, got {target.shape[0]}")
        if target.shape[1] != 1:
            raise ValueError(f"{path}: target channel dimension must be 1, got {target.shape[1]}")
        if obs.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(f"{path}: obs spatial size must be {self.image_size}x{self.image_size}")
        if target.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(f"{path}: target spatial size must be {self.image_size}x{self.image_size}")


class ToyFireRiskDataset(Dataset):
    """
    NPZ dataset generated from ToyFireEnv.

    obs: [N, k, C, 64, 64]
    target: [N, H, 1, 64, 64]
    """

    def __init__(
        self,
        npz_path: str | Path,
        max_samples: int | None = None,
    ) -> None:
        self.npz_path = Path(npz_path).expanduser()
        if not self.npz_path.exists():
            raise FileNotFoundError(f"ToyFireRiskDataset npz does not exist: {self.npz_path}")

        with np.load(self.npz_path) as data:
            if "obs" not in data or "target" not in data:
                raise KeyError(f"{self.npz_path} must contain 'obs' and 'target' arrays")
            obs = data["obs"].astype(np.float32, copy=False)
            target = data["target"].astype(np.float32, copy=False)

        self._validate_arrays(obs, target)
        if max_samples is not None:
            max_samples = max(1, min(int(max_samples), obs.shape[0]))
            obs = obs[:max_samples]
            target = target[:max_samples]

        self.obs = obs
        self.target = target

    def __len__(self) -> int:
        return int(self.obs.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "obs": torch.from_numpy(self.obs[idx]).float(),
            "target": torch.from_numpy(self.target[idx]).float(),
        }

    def high_risk_mask(self, threshold: float = 0.5) -> np.ndarray:
        return self.target.reshape(self.target.shape[0], -1).max(axis=1) > float(threshold)

    def _validate_arrays(self, obs: np.ndarray, target: np.ndarray) -> None:
        if obs.ndim != 5:
            raise ValueError(f"{self.npz_path}: obs must be [N,k,C,H,W], got {obs.shape}")
        if target.ndim != 5:
            raise ValueError(f"{self.npz_path}: target must be [N,H,1,H,W], got {target.shape}")
        if obs.shape[0] != target.shape[0]:
            raise ValueError(f"{self.npz_path}: obs/target sample count mismatch: {obs.shape[0]} vs {target.shape[0]}")
        if target.shape[2] != 1:
            raise ValueError(f"{self.npz_path}: target channel dimension must be 1, got {target.shape[2]}")
        if obs.shape[-2:] != (64, 64):
            raise ValueError(f"{self.npz_path}: obs spatial size must be 64x64, got {obs.shape[-2:]}")
        if target.shape[-2:] != (64, 64):
            raise ValueError(f"{self.npz_path}: target spatial size must be 64x64, got {target.shape[-2:]}")


def build_dataloader(
    config: dict[str, Any],
    split: str = "train",
    shuffle: bool | None = None,
) -> DataLoader:
    data_cfg = config.get("data", {})
    obs_window = int(config.get("obs_window", 4))
    horizon = int(config.get("horizon", 5))
    channels = int(config.get("channels", 6))
    image_size = int(config.get("image_size", 64))
    dataset_type = str(config.get("dataset_type", "dummy"))

    root_key = f"{split}_root"
    root = data_cfg.get(root_key)
    if dataset_type == "toy_npz":
        npz_key = f"{split}_npz"
        max_key = f"max_{split}_samples"
        npz_path = config.get(npz_key) or data_cfg.get(npz_key)
        if not npz_path:
            raise ValueError(f"dataset_type=toy_npz requires config key '{npz_key}'")
        dataset = ToyFireRiskDataset(npz_path=npz_path, max_samples=config.get(max_key))
    elif root:
        dataset: Dataset = FireRiskDataset(
            root=Path(root),
            obs_window=obs_window,
            horizon=horizon,
            image_size=image_size,
        )
    else:
        dataset = SyntheticFireRiskDataset(
            num_samples=int(data_cfg.get(f"{split}_samples", 128 if split == "train" else 32)),
            obs_window=obs_window,
            horizon=horizon,
            channels=channels,
            image_size=image_size,
            seed=int(data_cfg.get("seed", 42)) + (0 if split == "train" else 10_000),
        )

    sampler = None
    if split == "train" and dataset_type == "toy_npz" and isinstance(dataset, ToyFireRiskDataset):
        high_mask = dataset.high_risk_mask(float(config.get("high_risk_sample_threshold", 0.5)))
        high_count = int(high_mask.sum())
        total_count = int(high_mask.size)
        high_ratio_observed = high_count / max(1, total_count)
        print(
            "High-risk sample statistics: "
            f"split={split} total={total_count} high_risk={high_count} "
            f"ratio={high_ratio_observed:.6f} "
            f"oversampling_enabled={bool(config.get('use_high_risk_oversampling', False))}"
        )
    if (
        split == "train"
        and dataset_type == "toy_npz"
        and bool(config.get("use_high_risk_oversampling", False))
    ):
        if not isinstance(dataset, ToyFireRiskDataset):
            raise TypeError("High-risk oversampling requires ToyFireRiskDataset")
        high_mask = dataset.high_risk_mask(float(config.get("high_risk_sample_threshold", 0.5)))
        high_count = int(high_mask.sum())
        low_count = int(high_mask.size - high_count)
        min_fraction = float(config.get("high_risk_sample_min_fraction", 0.01))
        if high_count > 0 and low_count > 0 and high_count / max(1, high_mask.size) >= min_fraction:
            high_ratio = float(config.get("high_risk_sample_ratio", 0.5))
            high_ratio = min(0.99, max(0.01, high_ratio))
            weights = np.empty(high_mask.size, dtype=np.float64)
            weights[high_mask] = high_ratio / high_count
            weights[~high_mask] = (1.0 - high_ratio) / low_count
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(weights, dtype=torch.double),
                num_samples=len(dataset),
                replacement=True,
            )
            shuffle = False
            print(
                "Using high-risk oversampling: "
                f"high_samples={high_count} low_samples={low_count} target_ratio={high_ratio:.2f}"
            )
        else:
            print(
                "Warning: high-risk oversampling requested but skipped because "
                f"high_samples={high_count}, low_samples={low_count}, "
                f"min_fraction={min_fraction:.6f}."
            )

    if shuffle is None:
        shuffle = split == "train"

    return DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 8)),
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=bool(config.get("pin_memory", torch.cuda.is_available())),
        drop_last=bool(config.get("drop_last", False)),
    )


DummyFireRiskDataset = SyntheticFireRiskDataset
