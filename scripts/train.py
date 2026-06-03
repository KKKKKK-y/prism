from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch

from prism.config import load_config
from prism.datasets.dataset import build_dataloader
from prism.models.model import PRISMModel
from prism.trainers.trainer import PRISMTrainer, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PRISM fire risk forecasting model.")
    parser.add_argument("--config", type=Path, default=None, help="Path to YAML config.")
    parser.add_argument("--resume", type=Path, default=None, help="Path to checkpoint to resume.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--debug", action="store_true", help="Run a tiny one-epoch smoke training job.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.lr is not None:
        config["lr"] = args.lr
    if args.debug:
        config["epochs"] = 1
        config["batch_size"] = min(int(config.get("batch_size", 2)), 2)
        config["log_interval"] = 1
        if config.get("dataset_type", "dummy") == "toy_npz":
            config["max_train_samples"] = min(int(config.get("max_train_samples", 32)), 32)
            config["max_val_samples"] = min(int(config.get("max_val_samples", 16)), 16)
        else:
            config.setdefault("data", {})
            config["data"]["train_samples"] = min(int(config["data"].get("train_samples", 4)), 4)
            config["data"]["val_samples"] = min(int(config["data"].get("val_samples", 2)), 2)

    torch.manual_seed(int(config.get("seed", config.get("data", {}).get("seed", 42))))

    device = select_device()
    print(f"Using device: {device}")

    train_loader = build_dataloader(config, split="train")
    val_loader = build_dataloader(config, split="val", shuffle=False)

    model = PRISMModel(
        in_channels=int(config.get("channels", 6)),
        obs_window=int(config.get("obs_window", 4)),
        horizon=int(config.get("horizon", 5)),
        base_channels=int(config.get("base_channels", 32)),
        latent_channels=int(config.get("latent_channels", 128)),
        gru_layers=int(config.get("gru_layers", 1)),
        dropout_p=float(config.get("dropout_p", 0.1)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("lr", 1e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )

    run_name = datetime.now().strftime("prism_%Y%m%d_%H%M%S")
    output_dir = Path(config.get("output_dir", "prism/outputs")) / run_name

    trainer = PRISMTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=output_dir,
        device=device,
    )
    if args.resume is not None:
        trainer.resume_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
