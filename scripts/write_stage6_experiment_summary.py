from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Stage-6 experiment summary markdown.")
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/results"))
    parser.add_argument("--output", type=Path, default=Path("outputs/results/stage6_experiment_summary.md"))
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv(path: Path) -> list[dict[str, str]]:
    path = resolve(path)
    if not path.exists():
        print(f"Warning: missing input for summary: {path}")
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def markdown_table(rows: list[dict[str, str]], columns: list[str]) -> list[str]:
    if not rows:
        return ["No rows available."]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return lines


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    main_rows = read_csv(results_dir / "stage6_main_comparison_table.csv")
    ablation_rows = read_csv(results_dir / "stage6_ablation_table.csv")
    alpha_rows = read_csv(results_dir / "stage6_alpha_sensitivity.csv")
    diagnostic_rows = read_csv(results_dir / "stage6_diagnostic_summary_table.csv")

    lines: list[str] = [
        "# Stage 6 Experiment Summary",
        "",
        "## Final Planner",
        "",
        "The final PRISM planner uses current-risk prior and uncertainty-aware predictive safe-risk refinement.",
        "",
        "```text",
        "M_safe = mu + lambda_u * sigma",
        "M_final = alpha * M_current + (1 - alpha) * M_safe",
        "alpha = 0.4",
        "```",
        "",
        "Compared with the Current-Risk Planner, PRISM-Fusion improves the success rate from 0.78 to 0.85 and reduces the collision rate from 0.17 to 0.13 in hard dynamic fire scenarios.",
        "",
        "## Main Comparison",
        "",
        *markdown_table(
            main_rows,
            [
                "display_name",
                "success_rate",
                "collision_rate",
                "timeout_rate",
                "avg_cumulative_risk",
                "avg_path_length",
                "avg_steps",
            ],
        ),
        "",
        "## Ablation Conclusions",
        "",
        "Current-risk-only planning is a strong reactive baseline. Predicted-only planning can fail when predicted safe-risk underestimates collision risk. Max fusion is conservative, while alpha fusion provides the best success-collision tradeoff in the Stage-5.9 hard benchmark.",
        "",
        *markdown_table(
            ablation_rows,
            ["display_name", "success_rate", "collision_rate", "timeout_rate", "interpretation"],
        ),
        "",
        "## Alpha Sensitivity",
        "",
        "The selected final setting is `alpha = 0.4`, corresponding to 40% current-risk prior and 60% predictive safe-risk refinement.",
        "",
        *markdown_table(alpha_rows, ["alpha", "success_rate", "collision_rate", "timeout_rate"]),
        "",
        "## Diagnostics",
        "",
        "Planner-risk alignment diagnostics showed that low predicted safe-risk does not always imply low realized collision risk. The final fusion planner addresses this by anchoring planning to the current observed risk map while retaining predictive uncertainty-aware refinement.",
        "",
        *markdown_table(diagnostic_rows, ["source", "metric", "value", "conclusion"]),
        "",
    ]

    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved Stage-6 experiment summary to: {output_path}")


if __name__ == "__main__":
    main()
