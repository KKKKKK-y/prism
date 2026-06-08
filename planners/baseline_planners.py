from __future__ import annotations

from typing import Any

import torch

from prism.utils.uncertainty import mc_dropout_predict, propagate_uncertainty


STAGE5_METHODS = [
    "goal_greedy",
    "current_risk",
    "mean_risk",
    "prism_no_propagation",
    "prism_full",
]


def _model_obs(env_obs: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    obs = env_obs["obs"]
    if obs.ndim != 4:
        raise ValueError(f'env_obs["obs"] must be [k,C,H,W], got {tuple(obs.shape)}')
    return obs.unsqueeze(0).to(device)


def _require_model(method: str, model: torch.nn.Module | None) -> torch.nn.Module:
    if model is None:
        raise ValueError(f"Method {method!r} requires a loaded RiskPredictor model.")
    return model


def _check_planner_risk(planner_risk: torch.Tensor, horizon: int, height: int, width: int) -> torch.Tensor:
    if tuple(planner_risk.shape) != (horizon, 1, height, width):
        raise ValueError(
            "planner_risk must be "
            f"[{horizon},1,{height},{width}], got {tuple(planner_risk.shape)}"
        )
    return planner_risk.detach().cpu().float()


def build_planner_risk(
    method: str,
    env_obs: dict[str, torch.Tensor],
    model: torch.nn.Module | None,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Build the Stage-5 planner risk map for baseline and ablation methods.

    Returns:
        planner_risk: [H, 1, 64, 64]
        info: method-specific metadata
    """
    if method not in STAGE5_METHODS:
        raise ValueError(f"Unknown Stage-5 method {method!r}. Valid methods: {STAGE5_METHODS}")

    horizon = int(config.get("horizon", 5))
    risk_map = env_obs["risk_map"].detach().float()
    if risk_map.ndim != 2:
        raise ValueError(f'env_obs["risk_map"] must be [H,W], got {tuple(risk_map.shape)}')
    height, width = int(risk_map.shape[0]), int(risk_map.shape[1])
    info: dict[str, Any] = {"method": method}

    if method == "goal_greedy":
        planner_risk = torch.zeros(horizon, 1, height, width, dtype=torch.float32)
        return _check_planner_risk(planner_risk, horizon, height, width), info

    if method == "current_risk":
        planner_risk = risk_map.view(1, 1, height, width).repeat(horizon, 1, 1, 1)
        return _check_planner_risk(planner_risk, horizon, height, width), info

    model = _require_model(method, model).to(device)
    obs = _model_obs(env_obs, device)

    if method == "mean_risk":
        model.eval()
        with torch.no_grad():
            mu, _ = model(obs)
        planner_risk = mu[0]
        info["uses_uncertainty"] = False
        return _check_planner_risk(planner_risk, horizon, height, width), info

    num_samples = int(config.get("num_mc_samples", 5))
    lambda_u = float(config.get("lambda_u", 0.5))
    with torch.no_grad():
        mu_mean, sigma_mc, mu_samples = mc_dropout_predict(model, obs, num_samples=num_samples)

    if method == "prism_no_propagation":
        planner_risk = (mu_mean[0] + lambda_u * sigma_mc[0]).clamp(0.0, 1.0)
        info.update({"uses_uncertainty": True, "uncertainty": "sigma_mc", "num_mc_samples": num_samples})
        return _check_planner_risk(planner_risk, horizon, height, width), info

    sigma_prop, _ = propagate_uncertainty(mu_samples, alpha=float(config.get("uncertainty_alpha", 0.7)))
    planner_risk = (mu_mean[0] + lambda_u * sigma_prop[0]).clamp(0.0, 1.0)
    info.update({"uses_uncertainty": True, "uncertainty": "sigma_prop", "num_mc_samples": num_samples})
    return _check_planner_risk(planner_risk, horizon, height, width), info
