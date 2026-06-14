from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = PROJECT_ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from prism.scripts.sweep_hard_planner_params import sort_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Stage-5.2 hard planner diagnostics.")
    parser.add_argument(
        "--diagnostics",
        type=Path,
        default=Path("outputs/results/stage5_2_hard_planner_diagnostics.csv"),
    )
    parser.add_argument(
        "--sweep",
        type=Path,
        default=Path("outputs/results/stage5_2_hard_planner_sweep.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/results/stage5_2_diagnostic_summary.txt"),
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
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


def format_float(value: float, digits: int = 6) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def group_by(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return dict(grouped)


def last_rows_by_episode(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["episode"]))].append(row)
    last_rows = []
    for _, episode_rows in grouped.items():
        episode_rows.sort(key=lambda item: int(float(item["step"])))
        last_rows.append(episode_rows[-1])
    return last_rows


def classify_prism_full_failure(episode_rows: list[dict[str, str]]) -> str:
    if not episode_rows:
        return "unknown"
    episode_rows.sort(key=lambda item: int(float(item["step"])))
    last = episode_rows[-1]
    if int(float(last["success"])) == 1:
        return "success"

    zero_feasible_share = mean([1.0 if as_float(row["num_feasible"]) <= 0 else 0.0 for row in episode_rows])
    approach_ratios = []
    for row in episode_rows:
        distance = max(as_float(row["distance_to_goal"]), 1e-6)
        approach_ratios.append(as_float(row["selected_final_distance"]) / distance)
    avg_approach_ratio = mean(approach_ratios)
    avg_selected_risk = mean([as_float(row["selected_traj_risk"]) for row in episode_rows])
    avg_safe_p95 = mean([as_float(row["safe_risk_p95"]) for row in episode_rows])
    collision = int(float(last["collision"])) == 1

    if zero_feasible_share >= 0.25:
        return "no feasible trajectory"
    if avg_approach_ratio >= 0.95:
        return "selected trajectory not approaching goal"
    if avg_selected_risk >= 1.4 or avg_safe_p95 >= 0.75:
        return "risk too high"
    if collision and avg_selected_risk < 1.4:
        return "collision despite low predicted risk"
    return "mixed planner scoring issue"


def summarize_methods(rows: list[dict[str, str]]) -> list[str]:
    lines: list[str] = []
    by_method = group_by(rows, "method")
    lines.append("Method Diagnostics")
    lines.append("------------------")
    lines.append(
        "method avg_feasible_ratio avg_selected_traj_risk avg_selected_final_distance "
        "avg_num_feasible zero_feasible_step_rate"
    )
    for method in sorted(by_method):
        method_rows = by_method[method]
        avg_feasible = mean([as_float(row["feasible_ratio"]) for row in method_rows])
        avg_selected_risk = mean([as_float(row["selected_traj_risk"]) for row in method_rows])
        avg_selected_distance = mean([as_float(row["selected_final_distance"]) for row in method_rows])
        avg_num_feasible = mean([as_float(row["num_feasible"]) for row in method_rows])
        zero_feasible = mean([1.0 if as_float(row["num_feasible"]) <= 0 else 0.0 for row in method_rows])
        lines.append(
            f"{method} {format_float(avg_feasible)} {format_float(avg_selected_risk)} "
            f"{format_float(avg_selected_distance)} {format_float(avg_num_feasible)} {format_float(zero_feasible)}"
        )
    lines.append("")
    return lines


def summarize_prism_full(rows: list[dict[str, str]]) -> tuple[list[str], Counter[str]]:
    lines: list[str] = []
    prism_rows = [row for row in rows if row["method"] == "prism_full"]
    if not prism_rows:
        return ["Prism Full Diagnostics", "----------------------", "No prism_full rows found.", ""], Counter()

    safe_mean = mean([as_float(row["safe_risk_mean"]) for row in prism_rows])
    safe_p95 = mean([as_float(row["safe_risk_p95"]) for row in prism_rows])
    safe_max = mean([as_float(row["safe_risk_max"]) for row in prism_rows])
    avg_feasible = mean([as_float(row["feasible_ratio"]) for row in prism_rows])
    avg_selected_risk = mean([as_float(row["selected_traj_risk"]) for row in prism_rows])
    avg_selected_distance = mean([as_float(row["selected_final_distance"]) for row in prism_rows])
    zero_feasible_step_rate = mean([1.0 if as_float(row["num_feasible"]) <= 0 else 0.0 for row in prism_rows])

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in prism_rows:
        grouped[str(row["episode"])].append(row)
    failure_reasons = Counter()
    for episode_rows in grouped.values():
        reason = classify_prism_full_failure(episode_rows)
        if reason != "success":
            failure_reasons[reason] += 1

    final_rows = last_rows_by_episode(prism_rows)
    success_rate = mean([as_float(row["success"]) for row in final_rows])
    collision_rate = mean([as_float(row["collision"]) for row in final_rows])

    lines.append("Prism Full Diagnostics")
    lines.append("----------------------")
    lines.append(f"avg_feasible_ratio: {format_float(avg_feasible)}")
    lines.append(f"avg_selected_traj_risk: {format_float(avg_selected_risk)}")
    lines.append(f"avg_selected_final_distance: {format_float(avg_selected_distance)}")
    lines.append(
        "safe_risk_mean/p95/max: "
        f"{format_float(safe_mean)} / {format_float(safe_p95)} / {format_float(safe_max)}"
    )
    lines.append(f"zero_feasible_step_rate: {format_float(zero_feasible_step_rate)}")
    lines.append(f"diagnostic_success_rate: {format_float(success_rate)}")
    lines.append(f"diagnostic_collision_rate: {format_float(collision_rate)}")
    lines.append("failure_reason_counts:")
    if failure_reasons:
        for reason, count in failure_reasons.most_common():
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- none: 0")
    lines.append("")
    return lines, failure_reasons


def summarize_sweep(rows: list[dict[str, str]]) -> list[str]:
    lines: list[str] = []
    lines.append("Sweep Top 10")
    lines.append("------------")
    lines.append(
        "rank lambda_u risk_threshold_delta goal_weight trajectory_noise_scale "
        "num_trajectories success_rate collision_rate avg_cumulative_risk avg_feasible_ratio"
    )
    sorted_rows = sorted(rows, key=sort_key)
    for rank, row in enumerate(sorted_rows[:10], start=1):
        lines.append(
            f"{rank} "
            f"{format_float(as_float(row['lambda_u']), 3)} "
            f"{format_float(as_float(row['risk_threshold_delta']), 3)} "
            f"{format_float(as_float(row['goal_weight']), 3)} "
            f"{format_float(as_float(row['trajectory_noise_scale']), 3)} "
            f"{int(float(row['num_trajectories']))} "
            f"{format_float(as_float(row['success_rate']))} "
            f"{format_float(as_float(row['collision_rate']))} "
            f"{format_float(as_float(row['avg_cumulative_risk']))} "
            f"{format_float(as_float(row['avg_feasible_ratio']))}"
        )
    lines.append("")
    return lines


def recommendation(failure_reasons: Counter[str], sweep_rows: list[dict[str, str]]) -> list[str]:
    lines = ["Recommendation", "--------------"]
    top = sorted(sweep_rows, key=sort_key)[0] if sweep_rows else None
    dominant = failure_reasons.most_common(1)[0][0] if failure_reasons else "none"
    if dominant == "no feasible trajectory":
        lines.append("Primary next step: tune planner parameters, especially relax delta, reduce lambda_u, and add candidates.")
    elif dominant == "selected trajectory not approaching goal":
        lines.append("Primary next step: tune planner scoring, especially increase goal_weight/progress_weight.")
    elif dominant == "collision despite low predicted risk":
        lines.append(
            "Primary next step: inspect for a planner bug in trajectory risk sampling, horizon alignment, or [x,y] vs [y,x]."
        )
    elif dominant == "risk too high":
        lines.append("Primary next step: reduce uncertainty penalty and inspect sigma scale before any retraining.")
    else:
        lines.append("Primary next step: compare top sweep settings with default, then inspect scoring traces before retraining.")

    if top is not None:
        lines.append(
            "Best sweep row: "
            f"lambda_u={top['lambda_u']}, delta={top['risk_threshold_delta']}, "
            f"goal_weight={top['goal_weight']}, noise={top['trajectory_noise_scale']}, "
            f"num_trajectories={top['num_trajectories']}, "
            f"success={top['success_rate']}, collision={top['collision_rate']}."
        )
    lines.append("Do not continue Level C until the planner diagnosis is resolved.")
    lines.append("")
    return lines


def main() -> None:
    args = parse_args()
    diagnostic_rows = read_csv(args.diagnostics)
    sweep_rows = read_csv(args.sweep)

    lines: list[str] = []
    lines.append("PRISM Stage-5.2 Hard Planner Diagnostic Summary")
    lines.append("===============================================")
    lines.append(f"diagnostic_rows: {len(diagnostic_rows)}")
    lines.append(f"sweep_rows: {len(sweep_rows)}")
    lines.append("")
    lines.extend(summarize_methods(diagnostic_rows))
    prism_lines, failure_reasons = summarize_prism_full(diagnostic_rows)
    lines.extend(prism_lines)
    lines.extend(summarize_sweep(sweep_rows))
    lines.extend(recommendation(failure_reasons, sweep_rows))

    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved Stage-5.2 diagnostic summary to: {output}")


if __name__ == "__main__":
    main()
