#!/usr/bin/env python3
"""End-to-end EIS format choice plus allocation workflow.

The experiment links the two managerial steps in the paper:

1. Choose a feedback format using EIS(Q,S)=beta(S)*Q*alpha(Q,S).
2. Convert that format into raw response supply and item-level information,
   then allocate the resulting responses before evaluating the final top-K gap.

The candidate formats and completion scenarios match the EIS validation
exercise. The allocation policies reuse the transparent rules from the main
synthetic allocation benchmark.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

import numpy as np

from eis_format_validation import FORMATS, SCENARIOS, kappa
from synthetic_allocation_benchmarks import ar1_cov, choose_item, update_gaussian


ALLOCATION_POLICIES = ("uniform", "cutoff_only", "cutoff_x_uncertainty")
RICH_FORMAT_LABEL = "5-point, Q=7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=60)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--rho", type=float, default=0.6)
    parser.add_argument("--T", type=int, default=30)
    parser.add_argument("--worlds", type=int, default=100)
    parser.add_argument("--warm-start", type=int, default=5)
    parser.add_argument("--warm-start-scale", type=int, choices=[2, 5], default=5)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument(
        "--output",
        default="eis_allocation_workflow_summary.csv",
        help="CSV path for the workflow-level summary.",
    )
    parser.add_argument(
        "--raw-output",
        default="eis_allocation_workflow_raw.csv",
        help="CSV path for one row per scenario, world, and workflow.",
    )
    parser.add_argument(
        "--format-output",
        default="eis_allocation_workflow_format_raw.csv",
        help="CSV path for one row per scenario, world, format, and allocation policy.",
    )
    return parser.parse_args()


def scalar_se(values: np.ndarray) -> float:
    if len(values) <= 1:
        return float("nan")
    return float(values.std(ddof=1) / math.sqrt(len(values)))


def format_budget(customers: int, questions: int, completion_rate: float) -> int:
    return int(round(customers * questions * completion_rate))


def choose_by_score(scores: dict[str, float]) -> str:
    best_score = max(scores.values())
    best_labels = [label for label, score in scores.items() if np.isclose(score, best_score)]
    return sorted(best_labels)[0]


def warm_start_state(
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
    w_true: np.ndarray,
    warm_kappas: np.ndarray,
    warm_start: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs_var = 1.0 / np.clip(warm_kappas, 1e-8, None)
    mean = prior_mean.copy()
    cov = prior_cov.copy()
    counts = np.zeros(len(prior_mean), dtype=int)
    for item in range(len(prior_mean)):
        for _ in range(warm_start):
            y = w_true[item] + rng.normal(0.0, math.sqrt(obs_var[item]))
            counts[item] += 1
            mean, cov = update_gaussian(mean, cov, item, y, obs_var[item])
    return mean, cov, counts


def run_allocation(
    policy: str,
    initial_mean: np.ndarray,
    initial_cov: np.ndarray,
    initial_counts: np.ndarray,
    w_true: np.ndarray,
    kappas: np.ndarray,
    budget: int,
    top_k: int,
    rng: np.random.Generator,
) -> float:
    obs_var = 1.0 / np.clip(kappas, 1e-8, None)
    oracle_sum = float(np.sort(w_true)[-top_k:].sum())
    mean = initial_mean.copy()
    cov = initial_cov.copy()
    counts = initial_counts.copy()
    internal_policy = "even" if policy == "uniform" else policy

    for _ in range(max(0, int(budget))):
        item = choose_item(internal_policy, mean, cov, counts, kappas, obs_var, top_k, rng)
        y = w_true[item] + rng.normal(0.0, math.sqrt(obs_var[item]))
        counts[item] += 1
        mean, cov = update_gaussian(mean, cov, item, y, obs_var[item])

    chosen = np.argpartition(mean, -top_k)[-top_k:]
    return oracle_sum - float(w_true[chosen].sum())


def summarize_workflows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["scenario"]), str(row["workflow"])), []).append(row)

    by_scenario_world = {
        (str(row["scenario"]), int(row["world"]), str(row["workflow"])): row
        for row in rows
    }

    summary_rows: list[dict[str, object]] = []
    for (scenario, workflow), group in sorted(grouped.items()):
        gaps = np.array([float(row["gap"]) for row in group], dtype=float)
        budgets = np.array([float(row["budget"]) for row in group], dtype=float)
        regrets = np.array(
            [float(row["regret_vs_oracle_same_allocation"]) for row in group], dtype=float
        )
        selected_formats = [str(row["selected_format"]) for row in group]
        mode_format, mode_count = Counter(selected_formats).most_common(1)[0]

        improvements_vs_eis_uniform = []
        improvements_vs_most_answers = []
        wins_vs_eis_uniform = []
        wins_vs_most_answers = []
        for row in group:
            key_base = (scenario, int(row["world"]), "EIS + uniform")
            key_answers = (scenario, int(row["world"]), "Most answers + cutoff-only")
            base_gap = float(by_scenario_world[key_base]["gap"])
            answers_gap = float(by_scenario_world[key_answers]["gap"])
            gap = float(row["gap"])
            improvements_vs_eis_uniform.append(base_gap - gap)
            improvements_vs_most_answers.append(answers_gap - gap)
            wins_vs_eis_uniform.append(float(gap < base_gap))
            wins_vs_most_answers.append(float(gap < answers_gap))

        imp_uniform = np.array(improvements_vs_eis_uniform, dtype=float)
        imp_answers = np.array(improvements_vs_most_answers, dtype=float)

        summary_rows.append(
            {
                "scenario": scenario,
                "workflow": workflow,
                "format_rule": group[0]["format_rule"],
                "allocation_policy": group[0]["allocation_policy"],
                "worlds": len(group),
                "modal_selected_format": mode_format,
                "modal_selected_format_share": mode_count / len(group),
                "mean_budget": budgets.mean(),
                "mean_gap": gaps.mean(),
                "gap_se": scalar_se(gaps),
                "mean_regret_vs_oracle_same_allocation": regrets.mean(),
                "regret_vs_oracle_same_allocation_se": scalar_se(regrets),
                "format_hit_rate_vs_oracle_same_allocation": np.mean(
                    [
                        str(row["selected_format"])
                        == str(row["oracle_format_same_allocation"])
                        for row in group
                    ]
                ),
                "mean_improvement_vs_eis_uniform": imp_uniform.mean(),
                "improvement_vs_eis_uniform_se": scalar_se(imp_uniform),
                "win_rate_vs_eis_uniform": np.mean(wins_vs_eis_uniform),
                "mean_improvement_vs_most_answers_cutoff": imp_answers.mean(),
                "improvement_vs_most_answers_cutoff_se": scalar_se(imp_answers),
                "win_rate_vs_most_answers_cutoff": np.mean(wins_vs_most_answers),
            }
        )
    return summary_rows


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    raw_output_path = Path(args.raw_output)
    format_output_path = Path(args.format_output)
    master_rng = np.random.default_rng(args.seed)
    gh_x, gh_w = np.polynomial.hermite.hermgauss(31)
    prior_cov = ar1_cov(args.L, args.rho)

    format_rows: list[dict[str, object]] = []
    workflow_rows: list[dict[str, object]] = []

    for scenario_name, alpha_map in SCENARIOS.items():
        for world in range(args.worlds):
            world_seed = int(master_rng.integers(0, 2**32 - 1))
            world_rng = np.random.default_rng(world_seed)
            prior_mean = world_rng.normal(0.0, 1.0, size=args.L)
            w_true = world_rng.multivariate_normal(prior_mean, prior_cov)

            kappas_by_scale = {
                scale: np.array([kappa(float(mu), scale, gh_x, gh_w) for mu in prior_mean])
                for scale in (2, 5)
            }
            betas = {
                scale: float(kappas_for_scale.mean())
                for scale, kappas_for_scale in kappas_by_scale.items()
            }
            warm_mean, warm_cov, warm_counts = warm_start_state(
                prior_mean=prior_mean,
                prior_cov=prior_cov,
                w_true=w_true,
                warm_kappas=kappas_by_scale[args.warm_start_scale],
                warm_start=args.warm_start,
                rng=world_rng,
            )

            eis_scores = {}
            raw_answer_scores = {}
            completion_scores = {}
            budgets = {}
            for label, questions, scale in FORMATS:
                completion = alpha_map[label]
                eis_scores[label] = betas[scale] * questions * completion
                raw_answer_scores[label] = questions * completion
                completion_scores[label] = completion
                budgets[label] = format_budget(args.T, questions, completion)

            eis_format = choose_by_score(eis_scores)
            most_answers_format = choose_by_score(raw_answer_scores)
            highest_completion_format = choose_by_score(completion_scores)
            richest_format = RICH_FORMAT_LABEL

            format_policy_gaps: dict[tuple[str, str], float] = {}
            format_meta: dict[str, tuple[int, int, float]] = {}
            for label, questions, scale in FORMATS:
                format_meta[label] = (questions, scale, alpha_map[label])
                for policy in ALLOCATION_POLICIES:
                    policy_seed = int(world_rng.integers(0, 2**32 - 1))
                    policy_rng = np.random.default_rng(policy_seed)
                    gap = run_allocation(
                        policy=policy,
                        initial_mean=warm_mean,
                        initial_cov=warm_cov,
                        initial_counts=warm_counts,
                        w_true=w_true,
                        kappas=kappas_by_scale[scale],
                        budget=budgets[label],
                        top_k=args.K,
                        rng=policy_rng,
                    )
                    format_policy_gaps[(label, policy)] = gap
                    format_rows.append(
                        {
                            "scenario": scenario_name,
                            "world": world,
                            "world_seed": world_seed,
                            "format": label,
                            "questions": questions,
                            "scale": scale,
                            "completion_rate": alpha_map[label],
                            "budget": budgets[label],
                            "beta_scale": betas[scale],
                            "eis_score": eis_scores[label],
                            "raw_answer_score": raw_answer_scores[label],
                            "allocation_policy": policy,
                            "gap": gap,
                            "eis_selected_format": eis_format,
                            "most_answers_format": most_answers_format,
                            "highest_completion_format": highest_completion_format,
                            "richest_longest_format": richest_format,
                            "selected_by_eis": int(label == eis_format),
                            "selected_by_most_answers": int(label == most_answers_format),
                            "selected_by_highest_completion": int(
                                label == highest_completion_format
                            ),
                            "selected_by_richest_longest": int(label == richest_format),
                        }
                    )

            oracle_by_policy = {
                policy: min(
                    FORMATS,
                    key=lambda fmt: format_policy_gaps[(fmt[0], policy)],
                )[0]
                for policy in ALLOCATION_POLICIES
            }
            oracle_gap_by_policy = {
                policy: format_policy_gaps[(oracle_by_policy[policy], policy)]
                for policy in ALLOCATION_POLICIES
            }

            workflow_specs = [
                ("EIS + cutoff-only", "EIS", eis_format, "cutoff_only"),
                (
                    "EIS + cutoff x uncertainty",
                    "EIS",
                    eis_format,
                    "cutoff_x_uncertainty",
                ),
                ("EIS + uniform", "EIS", eis_format, "uniform"),
                (
                    "Most answers + cutoff-only",
                    "most_answers",
                    most_answers_format,
                    "cutoff_only",
                ),
                (
                    "Richest/longest + cutoff-only",
                    "richest_longest",
                    richest_format,
                    "cutoff_only",
                ),
                (
                    "Highest completion + cutoff-only",
                    "highest_completion",
                    highest_completion_format,
                    "cutoff_only",
                ),
                (
                    "Oracle format + cutoff-only",
                    "oracle_format",
                    oracle_by_policy["cutoff_only"],
                    "cutoff_only",
                ),
                (
                    "Oracle format + cutoff x uncertainty",
                    "oracle_format",
                    oracle_by_policy["cutoff_x_uncertainty"],
                    "cutoff_x_uncertainty",
                ),
            ]

            for workflow, format_rule, selected_format, policy in workflow_specs:
                questions, scale, completion = format_meta[selected_format]
                gap = format_policy_gaps[(selected_format, policy)]
                oracle_format = oracle_by_policy[policy]
                oracle_gap = oracle_gap_by_policy[policy]
                workflow_rows.append(
                    {
                        "scenario": scenario_name,
                        "world": world,
                        "world_seed": world_seed,
                        "workflow": workflow,
                        "format_rule": format_rule,
                        "allocation_policy": policy,
                        "selected_format": selected_format,
                        "questions": questions,
                        "scale": scale,
                        "completion_rate": completion,
                        "budget": budgets[selected_format],
                        "gap": gap,
                        "oracle_format_same_allocation": oracle_format,
                        "oracle_gap_same_allocation": oracle_gap,
                        "regret_vs_oracle_same_allocation": gap - oracle_gap,
                        "eis_selected_format": eis_format,
                        "most_answers_format": most_answers_format,
                        "highest_completion_format": highest_completion_format,
                        "richest_longest_format": richest_format,
                    }
                )

    summary_rows = summarize_workflows(workflow_rows)

    summary_fields = [
        "scenario",
        "workflow",
        "format_rule",
        "allocation_policy",
        "worlds",
        "modal_selected_format",
        "modal_selected_format_share",
        "mean_budget",
        "mean_gap",
        "gap_se",
        "mean_regret_vs_oracle_same_allocation",
        "regret_vs_oracle_same_allocation_se",
        "format_hit_rate_vs_oracle_same_allocation",
        "mean_improvement_vs_eis_uniform",
        "improvement_vs_eis_uniform_se",
        "win_rate_vs_eis_uniform",
        "mean_improvement_vs_most_answers_cutoff",
        "improvement_vs_most_answers_cutoff_se",
        "win_rate_vs_most_answers_cutoff",
    ]
    raw_fields = [
        "scenario",
        "world",
        "world_seed",
        "workflow",
        "format_rule",
        "allocation_policy",
        "selected_format",
        "questions",
        "scale",
        "completion_rate",
        "budget",
        "gap",
        "oracle_format_same_allocation",
        "oracle_gap_same_allocation",
        "regret_vs_oracle_same_allocation",
        "eis_selected_format",
        "most_answers_format",
        "highest_completion_format",
        "richest_longest_format",
    ]
    format_fields = [
        "scenario",
        "world",
        "world_seed",
        "format",
        "questions",
        "scale",
        "completion_rate",
        "budget",
        "beta_scale",
        "eis_score",
        "raw_answer_score",
        "allocation_policy",
        "gap",
        "eis_selected_format",
        "most_answers_format",
        "highest_completion_format",
        "richest_longest_format",
        "selected_by_eis",
        "selected_by_most_answers",
        "selected_by_highest_completion",
        "selected_by_richest_longest",
    ]

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    with raw_output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows(workflow_rows)

    with format_output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=format_fields)
        writer.writeheader()
        writer.writerows(format_rows)

    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
