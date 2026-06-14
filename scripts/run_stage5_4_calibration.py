from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PRISM Stage-5.4 prediction-risk calibration diagnostics.")
    parser.add_argument("--config", type=Path, default=Path("configs/toy_train_hard_level_b.yaml"))
    parser.add_argument("--hard-config", type=Path, default=Path("configs/toy_eval_hard.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints_toy_hard_b/best.pt"))
    parser.add_argument("--episodes", type=int, default=50)
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def mean(values: list[float], *, ignore_nan: bool = True) -> float:
    if ignore_nan:
        values = [value for value in values if not math.isnan(value)]
    if not values:
        return float("nan")
    return sum(values) / len(values)


def fmt(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def group_by_horizon(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["horizon"])].append(row)
    return dict(sorted(grouped.items(), key=lambda item: int(float(item[0]))))


def write_summary(calibration_csv: Path, compare_csv: Path, output: Path) -> None:
    calibration_rows = read_csv(calibration_csv)
    compare_rows = read_csv(compare_csv)
    lines: list[str] = []
    lines.append("PRISM Stage-5.4 Prediction-Risk Calibration Summary")
    lines.append("====================================================")
    lines.append(f"calibration_rows: {len(calibration_rows)}")
    lines.append(f"current_vs_predicted_rows: {len(compare_rows)}")
    lines.append("")

    lines.append("Calibration Horizon Summary")
    lines.append("---------------------------")
    lines.append(
        "horizon mu_mae safe_risk_mae mu_corr safe_risk_corr "
        "recall_05 recall_07 sigma_error_corr mu_p95 safe_risk_p95 true_risk_p95"
    )
    for horizon, rows in group_by_horizon(calibration_rows).items():
        lines.append(
            f"{horizon} "
            f"{fmt(mean([as_float(row['mu_mae']) for row in rows]))} "
            f"{fmt(mean([as_float(row['safe_risk_mae']) for row in rows]))} "
            f"{fmt(mean([as_float(row['mu_corr']) for row in rows]))} "
            f"{fmt(mean([as_float(row['safe_risk_corr']) for row in rows]))} "
            f"{fmt(mean([as_float(row['recall_05']) for row in rows]))} "
            f"{fmt(mean([as_float(row['recall_07']) for row in rows]))} "
            f"{fmt(mean([as_float(row['sigma_error_corr']) for row in rows]))} "
            f"{fmt(mean([as_float(row['mu_p95']) for row in rows]))} "
            f"{fmt(mean([as_float(row['safe_risk_p95']) for row in rows]))} "
            f"{fmt(mean([as_float(row['true_risk_p95']) for row in rows]))}"
        )
    lines.append("")

    lines.append("Current vs Predicted Horizon Summary")
    lines.append("------------------------------------")
    lines.append(
        "horizon current_mae mu_mae safe_risk_mae current_corr mu_corr safe_risk_corr "
        "current_recall mu_recall safe_risk_recall"
    )
    for horizon, rows in group_by_horizon(compare_rows).items():
        lines.append(
            f"{horizon} "
            f"{fmt(mean([as_float(row['current_mae']) for row in rows]))} "
            f"{fmt(mean([as_float(row['mu_mae']) for row in rows]))} "
            f"{fmt(mean([as_float(row['safe_risk_mae']) for row in rows]))} "
            f"{fmt(mean([as_float(row['current_corr']) for row in rows]))} "
            f"{fmt(mean([as_float(row['mu_corr']) for row in rows]))} "
            f"{fmt(mean([as_float(row['safe_risk_corr']) for row in rows]))} "
            f"{fmt(mean([as_float(row['current_high_risk_recall']) for row in rows]))} "
            f"{fmt(mean([as_float(row['mu_high_risk_recall']) for row in rows]))} "
            f"{fmt(mean([as_float(row['safe_risk_high_risk_recall']) for row in rows]))}"
        )
    lines.append("")

    current_mae = mean([as_float(row["current_mae"]) for row in compare_rows])
    mu_mae = mean([as_float(row["mu_mae"]) for row in compare_rows])
    safe_mae = mean([as_float(row["safe_risk_mae"]) for row in compare_rows])
    current_recall = mean([as_float(row["current_high_risk_recall"]) for row in compare_rows])
    mu_recall = mean([as_float(row["mu_high_risk_recall"]) for row in compare_rows])
    safe_recall = mean([as_float(row["safe_risk_high_risk_recall"]) for row in compare_rows])
    sigma_error_corr = mean([as_float(row["sigma_error_corr"]) for row in calibration_rows])
    safe_p95 = mean([as_float(row["safe_risk_p95"]) for row in calibration_rows])
    true_p95 = mean([as_float(row["true_risk_p95"]) for row in calibration_rows])

    lines.append("Diagnostic Answers")
    lines.append("------------------")
    lines.append(f"overall_current_mae: {fmt(current_mae)}")
    lines.append(f"overall_mu_mae: {fmt(mu_mae)}")
    lines.append(f"overall_safe_risk_mae: {fmt(safe_mae)}")
    lines.append(f"overall_current_recall_05: {fmt(current_recall)}")
    lines.append(f"overall_mu_recall_05: {fmt(mu_recall)}")
    lines.append(f"overall_safe_risk_recall_05: {fmt(safe_recall)}")
    lines.append(f"overall_sigma_error_corr: {fmt(sigma_error_corr)}")
    lines.append(f"overall_safe_risk_p95: {fmt(safe_p95)}")
    lines.append(f"overall_true_risk_p95: {fmt(true_p95)}")
    lines.append(f"mu_better_than_current_mae: {mu_mae < current_mae}")
    lines.append(f"safe_risk_improves_recall_over_mu: {safe_recall > mu_recall}")
    lines.append(f"sigma_tracks_error: {sigma_error_corr > 0.10}")
    lines.append(f"safe_risk_underestimates_high_risk_p95: {safe_p95 < true_p95}")
    if current_mae < mu_mae and current_mae < safe_mae:
        lines.append("interpretation: current_risk is closer to true future risk than model predictions on this hard benchmark.")
    elif safe_mae < current_mae:
        lines.append("interpretation: safe_risk is closer to true future risk than current_risk on average.")
    else:
        lines.append("interpretation: model predictions are competitive in MAE, but recall/uncertainty still need inspection.")
    if safe_p95 < true_p95 or safe_recall < current_recall:
        lines.append("recommendation: diagnose target generation and hard training distribution before changing model size.")
    else:
        lines.append("recommendation: tune safe-risk/planner scoring before regenerating data.")
    lines.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved Stage-5.4 calibration summary to: {output}")


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError(f"--episodes must be positive, got {args.episodes}")

    calibration_csv = Path("outputs/results/stage5_4_prediction_calibration.csv")
    compare_csv = Path("outputs/results/stage5_4_current_vs_predicted.csv")
    summary_txt = Path("outputs/results/stage5_4_calibration_summary.txt")
    seed0_png = Path("outputs/visualizations/stage5_4_prediction_calibration_seed0.png")
    failure_png = Path("outputs/visualizations/stage5_4_prediction_calibration_failure.png")
    zip_path = Path("outputs/prism_stage5_4_calibration.zip")

    run_step(
        [
            sys.executable,
            "scripts/diagnose_prediction_calibration.py",
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
            "--output",
            str(calibration_csv),
        ]
    )
    run_step(
        [
            sys.executable,
            "scripts/compare_current_vs_predicted_risk.py",
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
            "--output",
            str(compare_csv),
        ]
    )
    run_step(
        [
            sys.executable,
            "scripts/visualize_prediction_calibration.py",
            "--config",
            str(args.config),
            "--hard-config",
            str(args.hard_config),
            "--checkpoint",
            str(args.checkpoint),
            "--seed",
            "0",
            "--search-limit",
            str(max(10, args.episodes)),
            "--device",
            args.device,
            "--output",
            str(seed0_png),
            "--failure-output",
            str(failure_png),
        ]
    )
    write_summary(PROJECT_ROOT / calibration_csv, PROJECT_ROOT / compare_csv, PROJECT_ROOT / summary_txt)
    run_step(
        [
            sys.executable,
            "scripts/package_results.py",
            "--output",
            str(zip_path),
            "--items",
            str(calibration_csv),
            str(compare_csv),
            str(summary_txt),
            str(seed0_png),
            str(failure_png),
            str(args.config),
            str(args.hard_config),
            "README.md",
            "--exclude-protected",
        ]
    )

    print("Stage-5.4 outputs:")
    print(f"stage5_4_prediction_calibration.csv: {calibration_csv}")
    print(f"stage5_4_current_vs_predicted.csv: {compare_csv}")
    print(f"stage5_4_calibration_summary.txt: {summary_txt}")
    print(f"stage5_4_prediction_calibration_seed0.png: {seed0_png}")
    print(f"stage5_4_prediction_calibration_failure.png: {failure_png}")
    print(f"prism_stage5_4_calibration.zip: {zip_path}")


if __name__ == "__main__":
    main()
