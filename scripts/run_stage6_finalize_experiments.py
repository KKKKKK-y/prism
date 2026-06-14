from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = PROJECT_ROOT.parent
PROTECTED_SUFFIXES = {".pt", ".pth", ".ckpt", ".npz", ".npy", ".zip"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize Stage-6 paper-ready PRISM result artifacts.")
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/results"))
    parser.add_argument("--visualization-dir", type=Path, default=Path("outputs/visualizations"))
    parser.add_argument("--output", type=Path, default=Path("outputs/prism_stage6_paper_ready_results.zip"))
    return parser.parse_args()


def project_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    pieces = [str(PROJECT_PARENT)]
    if existing:
        pieces.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pieces)
    return env


def run_command(args: list[str]) -> None:
    command = [sys.executable, *args]
    print(f"\n$ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=project_env(), check=True)


def verify_zip(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected Stage-6 zip was not created: {path}")
    with zipfile.ZipFile(path, "r") as zf:
        protected = [name for name in zf.namelist() if Path(name).suffix.lower() in PROTECTED_SUFFIXES]
    if protected:
        protected_list = "\n".join(protected)
        raise RuntimeError(f"Stage-6 zip contains protected files:\n{protected_list}")
    print(f"Verified protected suffix exclusion for: {path}")


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    visualization_dir = args.visualization_dir
    output = args.output

    run_command(["scripts/build_stage6_paper_tables.py", "--results-dir", str(results_dir)])
    run_command(
        [
            "scripts/plot_stage6_paper_figures.py",
            "--results-dir",
            str(results_dir),
            "--output-dir",
            str(visualization_dir),
        ]
    )
    run_command(
        [
            "scripts/write_stage6_experiment_summary.py",
            "--results-dir",
            str(results_dir),
            "--output",
            str(results_dir / "stage6_experiment_summary.md"),
        ]
    )

    items = [
        results_dir / "stage6_main_comparison_table.csv",
        results_dir / "stage6_ablation_table.csv",
        results_dir / "stage6_alpha_sensitivity.csv",
        results_dir / "stage6_diagnostic_summary_table.csv",
        results_dir / "stage6_experiment_summary.md",
        visualization_dir / "stage6_main_comparison.png",
        visualization_dir / "stage6_success_collision_tradeoff.png",
        visualization_dir / "stage6_alpha_sensitivity.png",
        visualization_dir / "stage6_ablation_comparison.png",
        Path("configs/prism_final_fusion.yaml"),
        Path("README.md"),
    ]
    run_command(
        [
            "scripts/package_results.py",
            "--output",
            str(output),
            "--exclude-protected",
            "--items",
            *[str(item) for item in items],
        ]
    )
    verify_zip(PROJECT_ROOT / output)


if __name__ == "__main__":
    main()
