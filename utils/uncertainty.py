from __future__ import annotations

import torch
from torch import nn


def enable_mc_dropout(model: nn.Module) -> None:
    """Enable only dropout layers while the rest of the model remains in eval mode."""
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


@torch.no_grad()
def mc_dropout_predict(
    model: nn.Module,
    obs: torch.Tensor,
    num_samples: int = 5,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run MC Dropout inference.

    obs: [B, k, C, 64, 64]
    mu_samples: [N, B, H, 1, 64, 64]
    mu_mean/sigma_mc: [B, H, 1, 64, 64]
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")

    model.eval()
    enable_mc_dropout(model)

    samples = []
    for _ in range(num_samples):
        mu, _ = model(obs)
        samples.append(mu)

    mu_samples = torch.stack(samples, dim=0)
    mu_mean = mu_samples.mean(dim=0)
    var_mc = (mu_samples - mu_mean.unsqueeze(0)).pow(2).mean(dim=0)
    sigma_mc = torch.sqrt(var_mc + eps)
    return mu_mean, sigma_mc, mu_samples


def propagate_uncertainty(
    mu_samples: torch.Tensor,
    alpha: float = 0.7,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Propagate MC variance over prediction horizons.

    mu_samples: [N, B, H, 1, 64, 64]
    sigma_prop/propagated_var: [B, H, 1, 64, 64]
    """
    if mu_samples.ndim != 6:
        raise ValueError(f"mu_samples must be [N,B,H,1,H,W], got {tuple(mu_samples.shape)}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    mu_mean = mu_samples.mean(dim=0)
    var_mc = (mu_samples - mu_mean.unsqueeze(0)).pow(2).mean(dim=0)
    propagated_var = torch.empty_like(var_mc)
    propagated_var[:, 0] = var_mc[:, 0]

    for h in range(1, var_mc.shape[1]):
        propagated_var[:, h] = alpha * propagated_var[:, h - 1] + (1.0 - alpha) * var_mc[:, h]

    sigma_prop = torch.sqrt(propagated_var + eps)
    return sigma_prop, propagated_var


def compute_safe_risk(
    mu_mean: torch.Tensor,
    sigma: torch.Tensor,
    lambda_u: float = 0.5,
    clamp: bool = True,
) -> torch.Tensor:
    """Compute safe_risk = mu_mean + lambda_u * sigma."""
    safe_risk = mu_mean + lambda_u * sigma
    if clamp:
        safe_risk = safe_risk.clamp(0.0, 1.0)
    return safe_risk
