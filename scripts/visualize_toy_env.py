from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prism.config import load_config
from prism.scripts.run_closed_loop import build_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize ToyFireEnv fire/smoke/risk dynamics.")
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"), help="Path to YAML config.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/visualizations/stage4_toy_env_dynamics.png"),
        help="Output image path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    env = build_env(config, seed=int(config.get("data", {}).get("seed", 42)))
    env.reset(seed=int(config.get("data", {}).get("seed", 42)))

    capture_steps = [0, 5, 10, 15, 20]
    captures = {}
    for step in range(max(capture_steps) + 1):
        if step in capture_steps:
            captures[step] = {
                "fire": env.fire_map.clone(),
                "smoke": env.smoke_map.clone(),
                "risk": env.risk_map.clone(),
            }
        if step < max(capture_steps):
            env.update_dynamics()

    rows = [("fire_map", "fire"), ("smoke_map", "smoke"), ("risk_map", "risk")]
    fig, axes = plt.subplots(3, 5, figsize=(16, 9), constrained_layout=True)
    for row_idx, (row_title, key) in enumerate(rows):
        for col_idx, step in enumerate(capture_steps):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(captures[step][key].numpy(), cmap="inferno", origin="upper", vmin=0.0, vmax=1.0)
            ax.set_title(f"{row_title}\nt={step}", fontsize=10)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved toy environment dynamics to: {output_path}")


if __name__ == "__main__":
    main()
