from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.block(x))


class CNNEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        latent_channels: int = 128,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_channels, base_channels, stride=1, dropout_p=dropout_p),
            ConvBlock(base_channels, base_channels * 2, stride=2, dropout_p=dropout_p),
            ConvBlock(base_channels * 2, base_channels * 4, stride=2, dropout_p=dropout_p),
            nn.Conv2d(base_channels * 4, latent_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(p=dropout_p) if dropout_p > 0.0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 1,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(hidden_channels, hidden_channels // 2, dropout_p=dropout_p),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(hidden_channels // 2, hidden_channels // 4, dropout_p=dropout_p),
            nn.Conv2d(hidden_channels // 4, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PRISMModel(nn.Module):
    """
    PRISM: CNN Encoder + GRU Temporal Module + Dual Decoder Heads.

    Input:
        obs: [B, obs_window, C, 64, 64]
    Output:
        mu:      [B, horizon, 1, 64, 64]
        log_var: [B, horizon, 1, 64, 64]
    """

    def __init__(
        self,
        in_channels: int = 6,
        obs_window: int = 4,
        horizon: int = 5,
        base_channels: int = 32,
        latent_channels: int = 128,
        gru_layers: int = 1,
        dropout_p: float = 0.1,
        log_var_min: float = -10.0,
        log_var_max: float = 5.0,
    ) -> None:
        super().__init__()
        self.obs_window = obs_window
        self.horizon = horizon
        self.latent_channels = latent_channels
        self.dropout_p = dropout_p
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max

        self.encoder = CNNEncoder(in_channels, base_channels, latent_channels, dropout_p=dropout_p)
        self.temporal_gru = nn.GRU(
            input_size=latent_channels,
            hidden_size=latent_channels,
            num_layers=gru_layers,
            batch_first=True,
        )
        self.future_embedding = nn.Embedding(horizon, latent_channels)
        self.mean_head = DecoderHead(
            latent_channels,
            hidden_channels=latent_channels,
            out_channels=1,
            dropout_p=dropout_p,
        )
        self.var_head = DecoderHead(
            latent_channels,
            hidden_channels=latent_channels,
            out_channels=1,
            dropout_p=dropout_p,
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if obs.ndim != 5:
            raise ValueError(f"obs must be [B,T,C,H,W], got {tuple(obs.shape)}")

        batch_size, time_steps, channels, height, width = obs.shape
        if time_steps != self.obs_window:
            raise ValueError(f"Expected obs_window={self.obs_window}, got {time_steps}")
        if (height, width) != (64, 64):
            raise ValueError(f"Expected spatial size 64x64, got {height}x{width}")

        x = obs.reshape(batch_size * time_steps, channels, height, width)
        encoded = self.encoder(x)
        _, latent_channels, latent_h, latent_w = encoded.shape

        encoded = encoded.view(batch_size, time_steps, latent_channels, latent_h, latent_w)
        tokens = encoded.mean(dim=(-2, -1))
        _, hidden = self.temporal_gru(tokens)
        state = hidden[-1]

        future_ids = torch.arange(self.horizon, device=obs.device)
        future_tokens = state.unsqueeze(1) + self.future_embedding(future_ids).unsqueeze(0)
        future_maps = future_tokens.view(batch_size * self.horizon, latent_channels, 1, 1)
        future_maps = future_maps.expand(-1, -1, latent_h, latent_w).contiguous()

        mu = self.mean_head(future_maps)
        log_var = self.var_head(future_maps).clamp(self.log_var_min, self.log_var_max)

        mu = mu.view(batch_size, self.horizon, 1, height, width)
        log_var = log_var.view(batch_size, self.horizon, 1, height, width)
        return mu, log_var


RiskPredictor = PRISMModel
