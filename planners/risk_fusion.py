from __future__ import annotations

import torch


FUSION_MODES = [
    "current_only",
    "predicted_only",
    "max_fusion",
    "alpha_fusion",
    "calibrated_predicted",
    "max_calibrated_fusion",
]


def _as_horizon_map(risk: torch.Tensor, horizon: int | None = None) -> torch.Tensor:
    risk = torch.as_tensor(risk, dtype=torch.float32).detach()
    if risk.ndim == 2:
        if horizon is None:
            raise ValueError("horizon is required when risk is [H,W]")
        return risk.view(1, 1, risk.shape[0], risk.shape[1]).repeat(horizon, 1, 1, 1)
    if risk.ndim == 3:
        return risk.unsqueeze(1)
    if risk.ndim == 4:
        if risk.shape[1] != 1:
            raise ValueError(f"risk channel dimension must be 1, got {tuple(risk.shape)}")
        return risk
    raise ValueError(f"risk must be [H,W], [T,H,W], or [T,1,H,W], got {tuple(risk.shape)}")


def build_fused_risk_map(
    current_risk: torch.Tensor,
    predicted_risk: torch.Tensor,
    mode: str,
    alpha: float = 0.5,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Build planner risk maps for Stage-5.8 current/predicted fusion.

    Trajectory coordinates are [x, y]. Risk map indexing is [y, x], where
    y is image row and x is image column.

    Args:
        current_risk: [64,64], [H,64,64], or [H,1,64,64]
        predicted_risk: [H,1,64,64] or [H,64,64]
        mode: one of FUSION_MODES
        alpha: current-risk weight for alpha_fusion
        scale: calibration multiplier for predicted risk

    Returns:
        planner_risk: [H,1,64,64], clamped to [0,1]
    """
    if mode not in FUSION_MODES:
        raise ValueError(f"Unknown fusion mode {mode!r}. Valid modes: {FUSION_MODES}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0,1], got {alpha}")

    predicted = _as_horizon_map(predicted_risk)
    horizon = int(predicted.shape[0])
    current = _as_horizon_map(current_risk, horizon=horizon)
    if tuple(current.shape) != tuple(predicted.shape):
        raise ValueError(f"current and predicted risk shapes must match, got {tuple(current.shape)} and {tuple(predicted.shape)}")

    scaled_predicted = predicted * float(scale)
    if mode == "current_only":
        planner_risk = current
    elif mode == "predicted_only":
        planner_risk = predicted
    elif mode == "max_fusion":
        planner_risk = torch.maximum(current, predicted)
    elif mode == "alpha_fusion":
        planner_risk = float(alpha) * current + (1.0 - float(alpha)) * predicted
    elif mode == "calibrated_predicted":
        planner_risk = scaled_predicted
    elif mode == "max_calibrated_fusion":
        planner_risk = torch.maximum(current, scaled_predicted)
    else:  # pragma: no cover
        raise AssertionError(f"Unhandled fusion mode: {mode}")

    return planner_risk.clamp(0.0, 1.0).detach().cpu().float()
