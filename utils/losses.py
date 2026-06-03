from __future__ import annotations

import torch
import torch.nn.functional as F


def gaussian_risk_loss(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
    uncertainty_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Stage 1/2 supervised risk prediction loss.

    Shapes are [B, H, 1, image_size, image_size] for all tensors.
    """
    pred_loss = F.mse_loss(mu, target)
    sq_error = (mu - target).pow(2)
    unc_loss = (sq_error / torch.exp(log_var) + log_var).mean()
    total_loss = pred_loss + uncertainty_weight * unc_loss
    return total_loss, {"pred_loss": pred_loss.detach(), "unc_loss": unc_loss.detach()}
