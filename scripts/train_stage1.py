from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from prism.config import load_config
from prism.datasets import DummyFireRiskDataset
from prism.models import RiskPredictor
from prism.trainers.trainer import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PRISM Stage-1 next-step risk map predictor.")
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"), help="Path to YAML config.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("prism/outputs/checkpoints"),
        help="Directory where best.pt will be saved.",
    )
    return parser.parse_args()


def build_stage1_dataset(config: dict[str, Any]) -> DummyFireRiskDataset:
    data_cfg = config.get("data", {})
    train_samples = int(data_cfg.get("train_samples", 128))
    val_samples = int(data_cfg.get("val_samples", 32))
    return DummyFireRiskDataset(
        num_samples=train_samples + val_samples,
        obs_window=int(config.get("obs_window", 4)),
        horizon=int(config.get("horizon", 5)),
        channels=int(config.get("channels", 6)),
        image_size=int(config.get("image_size", 64)),
        seed=int(data_cfg.get("seed", 42)),
    )


def split_dataset(dataset: Dataset, config: dict[str, Any]) -> tuple[Subset, Subset]:
    data_cfg = config.get("data", {})
    train_samples = int(data_cfg.get("train_samples", 128))
    val_samples = int(data_cfg.get("val_samples", 32))
    indices = list(range(train_samples + val_samples))
    return Subset(dataset, indices[:train_samples]), Subset(dataset, indices[train_samples:])


def build_loader(dataset: Dataset, config: dict[str, Any], shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 8)),
        shuffle=shuffle,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=bool(config.get("pin_memory", torch.cuda.is_available())),
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


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        obs = batch["obs"].to(device, non_blocking=True)
        target_next = batch["target"][:, 0].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        mu, _ = model(obs)
        pred_next = mu[:, 0]
        loss = F.mse_loss(pred_next, target_next)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0

    for batch in loader:
        obs = batch["obs"].to(device, non_blocking=True)
        target_next = batch["target"][:, 0].to(device, non_blocking=True)
        mu, _ = model(obs)
        pred_next = mu[:, 0]
        loss = F.mse_loss(pred_next, target_next)
        total_loss += loss.item()

    return total_loss / max(1, len(loader))


def save_best_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    epoch: int,
    train_loss: float,
    val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "stage": "stage1_next_step_risk_prediction",
        },
        path,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    torch.manual_seed(int(config.get("data", {}).get("seed", 42)))

    device = select_device()
    dataset = build_stage1_dataset(config)
    train_dataset, val_dataset = split_dataset(dataset, config)
    train_loader = build_loader(train_dataset, config, shuffle=True)
    val_loader = build_loader(val_dataset, config, shuffle=False)

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("lr", 1e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )

    epochs = int(config.get("epochs", 1))
    grad_clip_norm = float(config.get("grad_clip_norm", 1.0))
    checkpoint_path = args.checkpoint_dir.expanduser() / "best.pt"
    best_val_loss = float("inf")
    final_train_loss = float("nan")
    final_val_loss = float("nan")

    print(f"Using device: {device}")
    print(f"Saving best checkpoint to: {checkpoint_path}")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, grad_clip_norm)
        val_loss = validate(model, val_loader, device)
        final_train_loss = train_loss
        final_val_loss = val_loss

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_best_checkpoint(checkpoint_path, model, optimizer, config, epoch, train_loss, val_loss)

        print(
            f"epoch={epoch + 1}/{epochs} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"best_val_loss={best_val_loss:.6f}"
        )

    print(f"Best checkpoint saved to: {checkpoint_path}")
    print(f"Final train loss: {final_train_loss:.6f}")
    print(f"Final val loss: {final_val_loss:.6f}")
    print("Next: python scripts/visualize_prediction.py")


if __name__ == "__main__":
    main()
