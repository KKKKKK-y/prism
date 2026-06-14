from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:  # pragma: no cover - optional local dependency
    class SummaryWriter:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            print("Warning: tensorboard is not installed. TensorBoard logging is disabled.")

        def add_scalar(self, *args: object, **kwargs: object) -> None:
            return None

        def close(self) -> None:
            return None

from prism.utils.device import select_device
from prism.utils.losses import gaussian_risk_loss


class PRISMTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        config: dict[str, Any],
        output_dir: str | Path,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = Path(config["checkpoint_dir"]) if config.get("checkpoint_dir") else self.output_dir / "checkpoints"
        self.log_dir = Path(config["log_dir"]) if config.get("log_dir") else self.output_dir / "tensorboard"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or select_device()
        self.model.to(self.device)

        self.epochs = int(config.get("epochs", 100))
        self.grad_clip_norm = float(config.get("grad_clip_norm", 1.0))
        self.uncertainty_weight = float(config.get("uncertainty_weight", config.get("alpha_unc", 0.1)))
        self.use_weighted_loss = bool(config.get("use_weighted_loss", False))
        self.use_weighted_uncertainty_loss = bool(config.get("use_weighted_uncertainty_loss", False))
        self.high_risk_threshold = float(config.get("high_risk_threshold", 0.5))
        self.high_risk_weight = float(config.get("high_risk_weight", 5.0))
        self.weighting_mode = str(config.get("weighting_mode", "threshold"))
        self.hard_extra_weight = float(config.get("hard_extra_weight", 10.0))
        self.max_loss_weight = float(config.get("max_loss_weight", 100.0))
        self.use_amp = bool(config.get("amp", True)) and self.device.type == "cuda"
        self.scaler = self._build_grad_scaler()
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.training_log_path = self._resolve_training_log_path()
        self._initialize_training_log()

    def _build_grad_scaler(self) -> torch.amp.GradScaler:
        try:
            return torch.amp.GradScaler("cuda", enabled=self.use_amp)
        except TypeError:
            return torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _resolve_training_log_path(self) -> Path | None:
        if self.config.get("training_log_csv"):
            return Path(self.config["training_log_csv"])
        if self.config.get("dataset_type") == "toy_npz":
            return Path("outputs/results/stage4_toy_training_log.csv")
        return None

    def _initialize_training_log(self) -> None:
        if self.training_log_path is None or self.start_epoch > 0:
            return
        if self.config.get("resume") and self.training_log_path.exists():
            return
        self.training_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.training_log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "train_pred_loss",
                    "train_unc_loss",
                    "val_loss",
                    "val_pred_loss",
                    "val_unc_loss",
                    "learning_rate",
                ],
            )
            writer.writeheader()

    def prediction_loss(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return gaussian_risk_loss(
            mu,
            log_var,
            target,
            uncertainty_weight=self.uncertainty_weight,
            use_weighted_loss=self.use_weighted_loss,
            use_weighted_uncertainty_loss=self.use_weighted_uncertainty_loss,
            high_risk_threshold=self.high_risk_threshold,
            high_risk_weight=self.high_risk_weight,
            weighting_mode=self.weighting_mode,
            hard_extra_weight=self.hard_extra_weight,
            max_loss_weight=self.max_loss_weight,
        )

    def train(self) -> None:
        run_started_at = time.time()
        for epoch in range(self.start_epoch, self.epochs):
            train_metrics = self.train_one_epoch(epoch)
            self.log_epoch("train", train_metrics, epoch)

            val_metrics = None
            if self.val_loader is not None:
                val_metrics = self.validate(epoch)
                self.log_epoch("val", val_metrics, epoch)

            monitored_loss = val_metrics["loss"] if val_metrics is not None else train_metrics["loss"]
            is_best = monitored_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = monitored_loss
            self.save_checkpoint(epoch=epoch, is_best=is_best)
            self.log_training_csv(epoch, train_metrics, val_metrics)
            elapsed = time.time() - run_started_at
            completed_epochs = epoch + 1 - self.start_epoch
            total_epochs = max(1, self.epochs - self.start_epoch)
            eta = elapsed / max(1, completed_epochs) * max(0, total_epochs - completed_epochs)
            val_loss = float("nan") if val_metrics is None else val_metrics["loss"]
            print(
                f"phase=epoch_summary epoch={epoch + 1}/{self.epochs} "
                f"train_loss={train_metrics['loss']:.6f} val_loss={val_loss:.6f} "
                f"lr={self.optimizer.param_groups[0]['lr']:.6g} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )

        self.writer.close()

    def log_training_csv(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
    ) -> None:
        if self.training_log_path is None:
            return
        row = {
            "epoch": epoch + 1,
            "train_loss": train_metrics.get("loss", float("nan")),
            "train_pred_loss": train_metrics.get("pred_loss", float("nan")),
            "train_unc_loss": train_metrics.get("unc_loss", float("nan")),
            "val_loss": "" if val_metrics is None else val_metrics.get("loss", float("nan")),
            "val_pred_loss": "" if val_metrics is None else val_metrics.get("pred_loss", float("nan")),
            "val_unc_loss": "" if val_metrics is None else val_metrics.get("unc_loss", float("nan")),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }
        self.training_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.training_log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        totals = {"loss": 0.0, "pred_loss": 0.0, "unc_loss": 0.0}
        epoch_started_at = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            obs = batch["obs"].to(self.device, non_blocking=True)
            target = batch["target"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                mu, log_var = self.model(obs)
                loss, loss_parts = self.prediction_loss(mu, log_var, target)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_metrics = {
                "loss": loss.detach().item(),
                "pred_loss": loss_parts["pred_loss"].item(),
                "unc_loss": loss_parts["unc_loss"].item(),
            }
            for key, value in batch_metrics.items():
                totals[key] += value
                self.writer.add_scalar(f"train_step/{key}", value, self.global_step)

            self.global_step += 1
            if batch_idx % int(self.config.get("log_interval", 10)) == 0:
                elapsed = time.time() - epoch_started_at
                completed = batch_idx + 1
                eta = elapsed / max(1, completed) * max(0, len(self.train_loader) - completed)
                print(
                    f"phase=train epoch={epoch + 1}/{self.epochs} "
                    f"step={batch_idx + 1}/{len(self.train_loader)} "
                    f"train_loss={batch_metrics['loss']:.6f} "
                    f"val_loss=nan "
                    f"lr={self.optimizer.param_groups[0]['lr']:.6g} "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )

        return {key: value / max(1, len(self.train_loader)) for key, value in totals.items()}

    @torch.no_grad()
    def validate(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        totals = {"loss": 0.0, "pred_loss": 0.0, "unc_loss": 0.0}
        val_started_at = time.time()

        for batch in self.val_loader:
            obs = batch["obs"].to(self.device, non_blocking=True)
            target = batch["target"].to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                mu, log_var = self.model(obs)
                loss, loss_parts = self.prediction_loss(mu, log_var, target)

            totals["loss"] += loss.item()
            totals["pred_loss"] += loss_parts["pred_loss"].item()
            totals["unc_loss"] += loss_parts["unc_loss"].item()

        metrics = {key: value / max(1, len(self.val_loader)) for key, value in totals.items()}
        elapsed = time.time() - val_started_at
        print(
            f"phase=validation epoch={epoch + 1}/{self.epochs} "
            f"train_loss=nan val_loss={metrics['loss']:.6f} "
            f"lr={self.optimizer.param_groups[0]['lr']:.6g} "
            f"elapsed={elapsed:.1f}s eta=0.0s",
            flush=True,
        )
        return metrics

    def log_epoch(self, split: str, metrics: dict[str, float], epoch: int) -> None:
        for key, value in metrics.items():
            self.writer.add_scalar(f"{split}/{key}", value, epoch)

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> Path:
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "config": self.config,
        }
        last_path = self.checkpoint_dir / "last.pt"
        torch.save(checkpoint, last_path)
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pt")
        return last_path

    def resume_checkpoint(self, checkpoint_path: str | Path) -> None:
        path = Path(checkpoint_path).expanduser()
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.global_step = int(checkpoint.get("global_step", 0))
        self.best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        print(f"Resumed checkpoint from {path} at epoch {self.start_epoch}")
