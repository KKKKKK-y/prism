from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS = ["goal_greedy", "current_risk", "mean_risk", "prism_no_propagation", "prism_full"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PRISM Stage-5.2 hard planner diagnostics.")
    parser.add_argument("--config", type=Path, default=Path("configs/toy_train_hard_level_b.yaml"))
    parser.add_argument("--hard-config", type=Path, default=Path("configs/toy_eval_hard.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints_toy_hard_b/best.pt"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--sweep-episodes", type=int, default=30)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Device to use for model inference.",
    )
    return parser.parse_args()


def run_step(command: list[str]) -> None:
    print(f"\nRunning: {' '.join(command)}", flush=True)
    env = dict(os.environ)
    pythonpath = str(PROJECT_ROOT.parent)
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, env=env)


def package_stage5_2(output_zip: Path, items: list[Path]) -> None:
    output_path = output_zip if output_zip.is_absolute() else PROJECT_ROOT / output_zip
    output_path.parent.mkdir(parents=True, exist_ok=True)
    added: set[Path] = set()
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in items:
            source = item if item.is_absolute() else PROJECT_ROOT / item
            if not source.exists():
                print(f"Warning: missing Stage-5.2 package item, skipping: {item}")
                continue
            files = [source] if source.is_file() else sorted(path for path in source.rglob("*") if path.is_file())
            for file_path in files:
                try:
                    arcname = file_path.relative_to(PROJECT_ROOT)
                except ValueError:
                    arcname = Path(file_path.name)
                if arcname in added:
                    continue
                if file_path.suffix.lower() in {".pt", ".npz"}:
                    print(f"Warning: refusing to package protected file: {arcname}")
                    continue
                zf.write(file_path, arcname)
                added.add(arcname)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    try:
        display_path = output_path.relative_to(PROJECT_ROOT)
    except ValueError:
        display_path = output_path
    print(f"Zip size: {size_mb:.2f} MB")
    print(f"Packaged Stage-5.2 diagnostics to: {display_path}")


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError(f"--episodes must be positive, got {args.episodes}")
    if args.sweep_episodes <= 0:
        raise ValueError(f"--sweep-episodes must be positive, got {args.sweep_episodes}")

    diagnostics_csv = Path("outputs/results/stage5_2_hard_planner_diagnostics.csv")
    sweep_csv = Path("outputs/results/stage5_2_hard_planner_sweep.csv")
    summary_txt = Path("outputs/results/stage5_2_diagnostic_summary.txt")
    seed0_png = Path("outputs/visualizations/stage5_2_hard_episode_seed0.png")
    failure_png = Path("outputs/visualizations/stage5_2_hard_episode_failure_examples.png")
    zip_path = Path("outputs/prism_stage5_2_diagnostics.zip")

    print("Stage-5.2 diagnostic method list:", " ".join(METHODS))
    run_step(
        [
            sys.executable,
            "scripts/diagnose_hard_planner.py",
            "--config",
            str(args.config),
            "--hard-config",
            str(args.hard_config),
            "--checkpoint",
            str(args.checkpoint),
            "--episodes",
            str(args.episodes),
            "--device",
            args.device,
            "--methods",
            *METHODS,
            "--output",
            str(diagnostics_csv),
        ]
    )
    run_step(
        [
            sys.executable,
            "scripts/visualize_hard_episode.py",
            "--config",
            str(args.config),
            "--hard-config",
            str(args.hard_config),
            "--checkpoint",
            str(args.checkpoint),
            "--seed",
            "0",
            "--device",
            args.device,
            "--methods",
            *METHODS,
            "--output",
            str(seed0_png),
            "--find-failure",
            "--failure-method",
            "prism_full",
            "--failure-output",
            str(failure_png),
        ]
    )
    run_step(
        [
            sys.executable,
            "scripts/sweep_hard_planner_params.py",
            "--config",
            str(args.config),
            "--hard-config",
            str(args.hard_config),
            "--checkpoint",
            str(args.checkpoint),
            "--episodes",
            str(args.sweep_episodes),
            "--method",
            "prism_full",
            "--device",
            args.device,
            "--output",
            str(sweep_csv),
        ]
    )
    run_step(
        [
            sys.executable,
            "scripts/summarize_hard_diagnostics.py",
            "--diagnostics",
            str(diagnostics_csv),
            "--sweep",
            str(sweep_csv),
            "--output",
            str(summary_txt),
        ]
    )
    package_stage5_2(
        zip_path,
        [
            diagnostics_csv,
            sweep_csv,
            summary_txt,
            seed0_png,
            failure_png,
            args.hard_config,
            args.config,
            Path("README.md"),
        ],
    )

    print("Stage-5.2 outputs:")
    print(f"stage5_2_hard_planner_diagnostics.csv: {diagnostics_csv}")
    print(f"stage5_2_hard_planner_sweep.csv: {sweep_csv}")
    print(f"stage5_2_diagnostic_summary.txt: {summary_txt}")
    print(f"stage5_2_hard_episode_seed0.png: {seed0_png}")
    print(f"stage5_2_hard_episode_failure_examples.png: {failure_png}")
    print(f"prism_stage5_2_diagnostics.zip: {zip_path}")


if __name__ == "__main__":
    main()
