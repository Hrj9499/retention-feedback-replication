#!/usr/bin/env python3
"""Stress test for cutoff crowding in the synthetic allocation study.

The baseline allocation experiment draws prior means from a diffuse normal
distribution. This script instead controls how many plausible items are packed
near the shipment cutoff, because the paper's mechanism predicts that targeted
feedback should matter most when the last-in/first-out boundary is crowded.
As in the main synthetic allocation script, additional feedback uses the
Gaussian information-equivalent channel calibrated to ordered-probit Fisher
information rather than drawing categorical ratings directly.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from synthetic_allocation_benchmarks import (
    POLICIES,
    ar1_cov,
    expected_kappa,
    run_world,
    scalar_se,
    summarize,
)


SCENARIOS = (
    ("Separated cutoff", 0.40),
    ("Moderately crowded cutoff", 0.16),
    ("Crowded cutoff", 0.055),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=60)
    parser.add_argument("--K", type=int, default=6)
    parser.add_argument("--rho", type=float, default=0.6)
    parser.add_argument("--S", type=int, default=5)
    parser.add_argument("--worlds", type=int, default=100)
    parser.add_argument("--warm-start", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument(
        "--output",
        default="crowded_cutoff_summary.csv",
        help="CSV path for the summary table.",
    )
    parser.add_argument(
        "--raw-output",
        default="crowded_cutoff_raw.csv",
        help="CSV path for one row per scenario/world/budget/policy.",
    )
    return parser.parse_args()


def make_prior_mean(
    rng: np.random.Generator, size: int, k: int, cutoff_spacing: float
) -> np.ndarray:
    """Construct a prior mean vector with controlled spacing around rank K."""
    ranks = np.arange(size)
    sorted_mean = np.empty(size)

    top_end = max(k - 4, 0)
    crowded_start = max(k - 4, 0)
    crowded_end = min(k + 8, size)

    if top_end > 0:
        sorted_mean[:top_end] = np.linspace(1.8, 0.65, top_end)

    center = 0.0
    local_ranks = ranks[crowded_start:crowded_end]
    sorted_mean[crowded_start:crowded_end] = (
        center
        + cutoff_spacing * (k - local_ranks)
        + rng.normal(0.0, 0.08 * cutoff_spacing, size=len(local_ranks))
    )

    if crowded_end < size:
        sorted_mean[crowded_end:] = np.linspace(-0.55, -2.0, size - crowded_end)

    # Keep the marginal distribution from being mechanically sorted against the
    # AR(1) covariance index; the scenario is about boundary density, not item ID.
    return sorted_mean[rng.permutation(size)]


def local_cutoff_gap(prior_mean: np.ndarray, k: int) -> float:
    ordered = np.sort(prior_mean)[::-1]
    return float(ordered[k - 1] - ordered[k])


def main() -> None:
    args = parse_args()
    budgets = [10 * args.L, 30 * args.L]
    gh_x, gh_w = np.polynomial.hermite.hermgauss(31)
    prior_cov = ar1_cov(args.L, args.rho)
    master_rng = np.random.default_rng(args.seed)

    records: dict[str, dict[str, list[list[float]]]] = {
        scenario_name: {policy: [] for policy in POLICIES}
        for scenario_name, _ in SCENARIOS
    }
    cutoff_gaps: dict[str, list[float]] = {
        scenario_name: [] for scenario_name, _ in SCENARIOS
    }
    raw_rows = []

    for scenario_name, spacing in SCENARIOS:
        for world in range(args.worlds):
            world_seed = int(master_rng.integers(0, 2**32 - 1))
            world_rng = np.random.default_rng(world_seed)
            prior_mean = make_prior_mean(world_rng, args.L, args.K, spacing)
            cutoff_gap = local_cutoff_gap(prior_mean, args.K)
            w_true = world_rng.multivariate_normal(prior_mean, prior_cov)
            kappas = np.array(
                [expected_kappa(float(mu), gh_x, gh_w) for mu in prior_mean],
                dtype=float,
            )
            world_results = run_world(
                prior_mean=prior_mean,
                prior_cov=prior_cov,
                w_true=w_true,
                kappas=kappas,
                budgets=budgets,
                warm_start=args.warm_start,
                k=args.K,
                rng=world_rng,
            )
            cutoff_gaps[scenario_name].append(cutoff_gap)
            for policy in POLICIES:
                records[scenario_name][policy].append(world_results[policy])
                for budget, gap in zip(budgets, world_results[policy]):
                    raw_rows.append(
                        {
                            "scenario": scenario_name,
                            "world": world,
                            "world_seed": world_seed,
                            "prior_cutoff_gap": cutoff_gap,
                            "budget": budget,
                            "policy": policy,
                            "gap": gap,
                        }
                    )

    rows = []
    for scenario_name, _ in SCENARIOS:
        stacked = {
            policy: np.array(records[scenario_name][policy], dtype=float)
            for policy in POLICIES
        }
        means = {}
        ses = {}
        for policy, values in stacked.items():
            means[policy], ses[policy] = summarize(values)

        # Win rates are within-family best-gap rates, not pairwise wins versus even.
        stacked_cube = np.stack([stacked[policy] for policy in POLICIES], axis=2)
        min_by_world = stacked_cube.min(axis=2)
        win_rates = {
            policy: np.isclose(stacked[policy], min_by_world).mean(axis=0)
            for policy in POLICIES
        }
        cutoff_gap_arr = np.array(cutoff_gaps[scenario_name], dtype=float)

        for idx, budget in enumerate(budgets):
            even_gap = means["even"][idx]
            cutoff_gap = means["cutoff_only"][idx]
            combined_gap = means["cutoff_x_uncertainty"][idx]
            cutoff_improvement = stacked["even"][:, idx] - stacked["cutoff_only"][:, idx]
            combined_improvement = (
                stacked["even"][:, idx] - stacked["cutoff_x_uncertainty"][:, idx]
            )
            policy_means = {policy: means[policy][idx] for policy in POLICIES}
            best_policy = min(policy_means, key=policy_means.get)
            rows.append(
                {
                    "scenario": scenario_name,
                    "worlds": args.worlds,
                    "mean_prior_cutoff_gap": cutoff_gap_arr.mean(),
                    "budget": budget,
                    "even_mean": even_gap,
                    "even_se": ses["even"][idx],
                    "even_win_rate": win_rates["even"][idx],
                    "uncertainty_mean": means["uncertainty_only"][idx],
                    "uncertainty_se": ses["uncertainty_only"][idx],
                    "uncertainty_win_rate": win_rates["uncertainty_only"][idx],
                    "cutoff_mean": cutoff_gap,
                    "cutoff_se": ses["cutoff_only"][idx],
                    "cutoff_win_rate": win_rates["cutoff_only"][idx],
                    "combined_mean": combined_gap,
                    "combined_se": ses["cutoff_x_uncertainty"][idx],
                    "combined_win_rate": win_rates["cutoff_x_uncertainty"][idx],
                    "cutoff_pct_gap_reduction": 100.0
                    * (even_gap - cutoff_gap)
                    / max(even_gap, 1e-8),
                    "combined_pct_gap_reduction": 100.0
                    * (even_gap - combined_gap)
                    / max(even_gap, 1e-8),
                    "cutoff_improvement_vs_even_mean": cutoff_improvement.mean(),
                    "cutoff_improvement_vs_even_se": scalar_se(cutoff_improvement),
                    "combined_improvement_vs_even_mean": combined_improvement.mean(),
                    "combined_improvement_vs_even_se": scalar_se(combined_improvement),
                    "best_policy_by_mean": best_policy,
                }
            )

    raw_output_path = Path(args.raw_output)
    with raw_output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario",
                "world",
                "world_seed",
                "prior_cutoff_gap",
                "budget",
                "policy",
                "gap",
            ],
        )
        writer.writeheader()
        writer.writerows(raw_rows)

    output_path = Path(args.output)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario",
                "worlds",
                "mean_prior_cutoff_gap",
                "budget",
                "even_mean",
                "even_se",
                "even_win_rate",
                "uncertainty_mean",
                "uncertainty_se",
                "uncertainty_win_rate",
                "cutoff_mean",
                "cutoff_se",
                "cutoff_win_rate",
                "combined_mean",
                "combined_se",
                "combined_win_rate",
                "cutoff_pct_gap_reduction",
                "combined_pct_gap_reduction",
                "cutoff_improvement_vs_even_mean",
                "cutoff_improvement_vs_even_se",
                "combined_improvement_vs_even_mean",
                "combined_improvement_vs_even_se",
                "best_policy_by_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
