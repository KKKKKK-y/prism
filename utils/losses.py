from __future__ import annotations

import torch
import torch.nn.functional as F


def build_risk_weights(
    target: torch.Tensor,
    high_risk_threshold: float = 0.5,
    high_risk_weight: float = 5.0,
    weighting_mode: str = "threshold",
    hard_extra_weight: float = 10.0,
    max_loss_weight: float = 100.0,
) -> torch.Tensor:
    """Build per-pixel weights for high-risk aware supervision."""
    if weighting_mode == "threshold":
        weights = 1.0 + high_risk_weight * (target > high_risk_threshold).to(target.dtype)
    elif weighting_mode == "soft":
        weights = 1.0 + high_risk_weight * target
    elif weighting_mode == "hybrid":
        hard_mask = (target > high_risk_threshold).to(target.dtype)
        weights = 1.0 + high_risk_weight * target + hard_extra_weight * hard_mask
    else:
        raise ValueError(f"Unsupported weighting_mode: {weighting_mode!r}")
    return weights.clamp(max=max_loss_weight)


def gaussian_risk_loss(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
    uncertainty_weight: float = 0.1,
    use_weighted_loss: bool = False,
    use_weighted_uncertainty_loss: bool = False,
    high_risk_threshold: float = 0.5,
    high_risk_weight: float = 5.0,
    weighting_mode: str = "threshold",
    hard_extra_weight: float = 10.0,
    max_loss_weight: float = 100.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Stage 1/2 supervised risk prediction loss.

    Shapes are [B, H, 1, image_size, image_size] for all tensors.
    """
    sq_error = (mu - target).pow(2)
    weights = build_risk_weights(
        target=target,
        high_risk_threshold=high_risk_threshold,
        high_risk_weight=high_risk_weight,
        weighting_mode=weighting_mode,
        hard_extra_weight=hard_extra_weight,
        max_loss_weight=max_loss_weight,
    )

    if use_weighted_loss:
        pred_loss = (weights * sq_error).mean()
    else:
        pred_loss = F.mse_loss(mu, target)

    unc_terms = sq_error / torch.exp(log_var) + log_var
    if use_weighted_uncertainty_loss:
        unc_loss = (weights * unc_terms).mean()
    else:
        unc_loss = unc_terms.mean()

    total_loss = pred_loss + uncertainty_weight * unc_loss
    return total_loss, {
        "pred_loss": pred_loss.detach(),
        "unc_loss": unc_loss.detach(),
        "mean_weight": weights.mean().detach(),
    }
