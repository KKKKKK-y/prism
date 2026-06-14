from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DISPLAY_NAMES = {
    "goal_greedy": "Goal-Greedy Planner",
    "current_only": "Current-Risk Planner",
    "predicted_only": "Predictive-Risk Planner",
    "max_fusion": "Max-Fusion Planner",
    "alpha_fusion_0.4": "PRISM-Fusion (Ours)",
}

METHOD_ORDER = {
    "goal_greedy": 0,
    "current_only": 1,
    "predicted_only": 2,
    "max_fusion": 3,
    "alpha_fusion_0.4": 4,
}

MAIN_FIELDS = [
    "method",
    "display_name",
    "episodes",
    "success_rate",
    "collision_rate",
    "timeout_rate",
    "avg_cumulative_risk",
    "avg_path_length",
    "avg_steps",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage-6 paper-ready CSV tables from Stage-5 outputs.")
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/results"))
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv(path: Path) -> list[dict[str, str]]:
    path = resolve(path)
    if not path.exists():
        print(f"Warning: missing input, skipping: {path}")
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def to_float(value: str | float | int | None) -> float | None:
    if value is None or value == "" or str(value).upper() == "NA":
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if math.isnan(number):
        return None
    return number


def build_main_table(results_dir: Path) -> list[dict[str, Any]]:
    rows = read_csv(results_dir / "stage5_9_final_fusion_results.csv")
    if not rows:
        return []
    normalized = []
    for row in rows:
        method = row["method"]
        normalized.append(
            {
                "method": method,
                "display_name": DISPLAY_NAMES.get(method, method),
                "episodes": row.get("episodes", ""),
                "success_rate": row.get("success_rate", ""),
                "collision_rate": row.get("collision_rate", ""),
                "timeout_rate": row.get("timeout_rate", ""),
                "avg_cumulative_risk": row.get("avg_cumulative_risk", ""),
                "avg_path_length": row.get("avg_path_length", ""),
                "avg_steps": row.get("avg_steps", ""),
            }
        )
    normalized.sort(key=lambda item: METHOD_ORDER.get(item["method"], 99))
    write_csv(results_dir / "stage6_main_comparison_table.csv", normalized, MAIN_FIELDS)
    return normalized


def build_ablation_table(results_dir: Path, main_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not main_rows:
        print("Warning: main comparison table unavailable; skipping ablation table.")
        return []
    interpretations = {
        "current_only": "Current-risk prior is strong and reactive but lacks predictive refinement.",
        "predicted_only": "Predicted-only planning is brittle when predicted risk is under-calibrated.",
        "max_fusion": "Max fusion is more conservative but can still trail alpha fusion.",
        "alpha_fusion_0.4": "Final PRISM-Fusion balances current risk and predictive safe-risk refinement.",
    }
    rows = []
    for row in main_rows:
        method = row["method"]
        if method not in interpretations:
            continue
        rows.append(
            {
                **row,
                "ablation": method,
                "interpretation": interpretations[method],
            }
        )
    fields = ["ablation", *MAIN_FIELDS, "interpretation"]
    write_csv(results_dir / "stage6_ablation_table.csv", rows, fields)
    return rows


def build_alpha_sensitivity(results_dir: Path) -> list[dict[str, Any]]:
    rows = read_csv(results_dir / "stage5_8_fusion_all_checkpoints.csv")
    if not rows:
        return []
    selected = []
    for row in rows:
        if row.get("checkpoint_name") != "unweighted_hard_b":
            continue
        if row.get("mode") != "alpha_fusion":
            continue
        alpha = to_float(row.get("alpha"))
        if alpha is None:
            continue
        selected.append(
            {
                "checkpoint_name": row.get("checkpoint_name", ""),
                "mode": row.get("mode", ""),
                "alpha": alpha,
                "success_rate": row.get("success_rate", ""),
                "collision_rate": row.get("collision_rate", ""),
                "timeout_rate": row.get("timeout_rate", ""),
                "avg_cumulative_risk": row.get("avg_cumulative_risk", ""),
                "avg_path_length": row.get("avg_path_length", ""),
                "avg_steps": row.get("avg_steps", ""),
            }
        )
    if not selected:
        print("Warning: no unweighted_hard_b alpha_fusion rows found; skipping alpha sensitivity table.")
        return []
    selected.sort(key=lambda item: float(item["alpha"]))
    write_csv(
        results_dir / "stage6_alpha_sensitivity.csv",
        selected,
        [
            "checkpoint_name",
            "mode",
            "alpha",
            "success_rate",
            "collision_rate",
            "timeout_rate",
            "avg_cumulative_risk",
            "avg_path_length",
            "avg_steps",
        ],
    )
    return selected


def mean_numeric(rows: list[dict[str, str]], column: str) -> float | None:
    values = [to_float(row.get(column)) for row in rows]
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def best_by(rows: list[dict[str, str]], column: str, maximize: bool) -> dict[str, str] | None:
    candidates = []
    for row in rows:
        value = to_float(row.get(column))
        if value is not None:
            candidates.append((value, row))
    if not candidates:
        return None
    return sorted(candidates, key=lambda pair: pair[0], reverse=maximize)[0][1]


def build_diagnostic_summary(results_dir: Path) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []

    calibration_rows = read_csv(results_dir / "stage5_4_prediction_calibration.csv")
    if calibration_rows:
        diagnostics.extend(
            [
                {
                    "source": "stage5_4_prediction_calibration",
                    "metric": "mean_mu_mae",
                    "value": mean_numeric(calibration_rows, "mu_mae"),
                    "conclusion": "Prediction error remains meaningful in hard dynamic scenes.",
                },
                {
                    "source": "stage5_4_prediction_calibration",
                    "metric": "mean_safe_risk_mae",
                    "value": mean_numeric(calibration_rows, "safe_risk_mae"),
                    "conclusion": "Safe-risk calibration motivates retaining the current-risk prior.",
                },
                {
                    "source": "stage5_4_prediction_calibration",
                    "metric": "mean_recall_05",
                    "value": mean_numeric(calibration_rows, "recall_05"),
                    "conclusion": "High-risk recall diagnostics explain predicted-only planner failures.",
                },
            ]
        )

    sweep_rows = read_csv(results_dir / "stage5_7_high_risk_training_sweep.csv")
    if sweep_rows:
        best_recall = best_by(sweep_rows, "safe_recall_05", maximize=True)
        best_mae = best_by(sweep_rows, "overall_mae", maximize=False)
        if best_recall is not None:
            diagnostics.append(
                {
                    "source": "stage5_7_high_risk_training_sweep",
                    "metric": "best_safe_recall_05_config",
                    "value": best_recall.get("config_name", ""),
                    "conclusion": "Weighted training improves some diagnostics but is not selected as the final planner.",
                }
            )
        if best_mae is not None:
            diagnostics.append(
                {
                    "source": "stage5_7_high_risk_training_sweep",
                    "metric": "best_overall_mae_config",
                    "value": best_mae.get("config_name", ""),
                    "conclusion": "Prediction MAE alone does not determine closed-loop planner quality.",
                }
            )

    main_rows = read_csv(results_dir / "stage5_9_final_fusion_results.csv")
    current = next((row for row in main_rows if row.get("method") == "current_only"), None)
    ours = next((row for row in main_rows if row.get("method") == "alpha_fusion_0.4"), None)
    if current and ours:
        diagnostics.append(
            {
                "source": "stage5_9_final_fusion_results",
                "metric": "current_vs_alpha_fusion",
                "value": (
                    f"success {current['success_rate']} -> {ours['success_rate']}; "
                    f"collision {current['collision_rate']} -> {ours['collision_rate']}"
                ),
                "conclusion": "PRISM-Fusion improves success and reduces collision compared with current-risk-only planning.",
            }
        )

    if diagnostics:
        write_csv(
            results_dir / "stage6_diagnostic_summary_table.csv",
            diagnostics,
            ["source", "metric", "value", "conclusion"],
        )
    else:
        print("Warning: no diagnostic inputs found; skipping diagnostic summary table.")
    return diagnostics


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    main_rows = build_main_table(results_dir)
    build_ablation_table(results_dir, main_rows)
    build_alpha_sensitivity(results_dir)
    build_diagnostic_summary(results_dir)


if __name__ == "__main__":
    main()
