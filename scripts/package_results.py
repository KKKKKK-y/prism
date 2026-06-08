from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


DEFAULT_ITEMS = [
    Path("outputs/checkpoints_toy/best.pt"),
    Path("outputs/checkpoints_toy/last.pt"),
    Path("outputs/results/formal_run_summary.txt"),
    Path("outputs/results/stage5_baseline_results.csv"),
    Path("outputs/results/stage5_baseline_episode_results.csv"),
    Path("outputs/visualizations/stage5_baseline_comparison.png"),
    Path("outputs/results"),
    Path("outputs/visualizations"),
    Path("configs/toy_train.yaml"),
    Path("README.md"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package PRISM formal run results into a zip archive.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/prism_formal_results.zip"),
        help="Output zip path.",
    )
    return parser.parse_args()


def iter_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(item for item in path.rglob("*") if item.is_file())
    return []


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output_path = args.output if args.output.is_absolute() else project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    added: set[Path] = set()
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in DEFAULT_ITEMS:
            source = item if item.is_absolute() else project_root / item
            files = iter_files(source)
            if not files:
                print(f"Warning: missing result item, skipping: {item}")
                continue
            for file_path in files:
                try:
                    arcname = file_path.relative_to(project_root)
                except ValueError:
                    arcname = file_path.name
                if arcname in added:
                    continue
                zf.write(file_path, arcname)
                added.add(arcname)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    try:
        display_path = output_path.relative_to(project_root)
    except ValueError:
        display_path = output_path
    print(f"Zip size: {size_mb:.2f} MB")
    print(f"Packaged results to: {display_path}")


if __name__ == "__main__":
    main()
