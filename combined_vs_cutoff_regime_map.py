#!/usr/bin/env python3
"""Regime map for when uncertainty refines cutoff targeting.

The main allocation benchmark shows that cutoff-only targeting captures most of
the value in common warm-start settings. This experiment maps when the added
uncertainty term in the cutoff x uncertainty rule is useful. It varies boundary
crowding, initial information-stock heterogeneity near the cutoff, and signal
informativeness, then compares cutoff x uncertainty directly with cutoff-only.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from synthetic_allocation_benchmarks import (
    ar1_cov,
    boundary_weight,
    expected_kappa,
    scalar_se,
    total_variance_drop,
    update_gaussian,
)


POLICIES = ("cutoff_only", "cutoff_x_uncertainty")
DENSITY_SCENARIOS = (
    ("separated", 0.40),
    ("moderate", 0.16),
    ("crowded", 0.055),
)
INFO_STOCK_SCENARIOS = ("balanced", "heterogeneous")
INFORMATION_SCALE_SCENARIOS = (
    ("low_info", 0.65),
    ("baseline_info", 1.00),
    ("high_info", 1.50),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=60)
    parser.add_argument("--K", type=int, default=6)
    parser.add_argument("--rho", type=float, default=0.6)
    parser.add_argument("--worlds", type=int, default=100)
    parser.add_argument("--warm-start", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument(
        "--budgets",
        default="1,3,10,30",
        help="Comma-separated additional-response budgets as multiples of L.",
    )
    parser.add_argument(
        "--output",
        default="combined_vs_cutoff_regime_map_summary.csv",
        help="CSV path for the summary table.",
    )
    parser.add_argument(
        "--raw-output",
        default="combined_vs_cutoff_regime_map_raw.csv",
        help="CSV path for one row per regime, world, budget, and policy.",
    )
    return parser.parse_args()


def parse_budget_multiples(text: str) -> list[int]:
    multiples = [int(piece.strip()) for piece in text.split(",") if piece.strip()]
    if not multiples or any(m <= 0 for m in multiples):
        raise ValueError("--budgets must contain positive integer multiples of L.")
    return sorted(set(multiples))


def make_prior_mean(
    rng: np.random.Generator, size: int, k: int, cutoff_spacing: float
) -> np.ndarray:
    """Construct means with controlled spacing around the top-K boundary."""
    ranks = np.arange(size)
    sorted_mean = np.empty(size)

    top_end = max(k - 4, 0)
    crowded_start = max(k - 4, 0)
    crowded_end = min(k + 8, size)

    if top_end > 0:
        sorted_mean[:top_end] = np.linspace(1.8, 0.65, top_end)

    local_ranks = ranks[crowded_start:crowded_end]
    sorted_mean[crowded_start:crowded_end] = (
        cutoff_spacing * (k - local_ranks)
        + rng.normal(0.0, 0.08 * cutoff_spacing, size=len(local_ranks))
    )

    if crowded_end < size:
        sorted_mean[crowded_end:] = np.linspace(-0.55, -2.0, size - crowded_end)

    return sorted_mean[rng.permutation(size)]


def local_cutoff_gap(prior_mean: np.ndarray, k: int) -> float:
    ordered = np.sort(prior_mean)[::-1]
    return float(ordered[k - 1] - ordered[k])


def initial_counts(prior_mean: np.ndarray, k: int, warm_start: int, scenario: str) -> np.ndarray:
    counts = np.full(len(prior_mean), warm_start, dtype=int)
    if scenario == "balanced":
        return counts
    if scenario != "heterogeneous":
        raise ValueError(f"Unknown information-stock scenario: {scenario}")

    order = np.argsort(-prior_mean)
    obvious_top = order[: max(0, k - 3)]
    counts[obvious_top] = max(20, 6 * warm_start)

    boundary_band = order[max(0, k - 3) : min(len(prior_mean), k + 9)]
    high_count = max(60, 14 * warm_start)
    for offset, item in enumerate(boundary_band):
        counts[item] = high_count if offset % 2 == 0 else 1
    return counts


def choose_item(
    policy: str,
    mean: np.ndarray,
    cov: np.ndarray,
    counts: np.ndarray,
    obs_var: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> int:
    cutoff_score = boundary_weight(mean, k)
    uncertainty_score = total_variance_drop(cov, obs_var)

    if policy == "cutoff_only":
        scores = cutoff_score / (1.0 + 0.10 * counts)
    elif policy == "cutoff_x_uncertainty":
        scores = (
            cutoff_score
            * (0.15 + 0.85 * uncertainty_score / max(float(uncertainty_score.max()), 1e-12))
            / (1.0 + 0.05 * counts)
        )
    else:
        raise ValueError(f"Unknown policy: {policy}")

    best = np.flatnonzero(np.isclose(scores, scores.max()))
    return int(rng.choice(best))


def run_world(
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
    w_true: np.ndarray,
    kappas: np.ndarray,
    initial_count_scenario: str,
    budgets: list[int],
    warm_start: int,
    k: int,
    rng: np.random.Generator,
) -> dict[str, list[float]]:
    obs_var = 1.0 / np.clip(kappas, 1e-8, None)
    oracle_sum = float(np.sort(w_true)[-k:].sum())

    mean0 = prior_mean.copy()
    cov0 = prior_cov.copy()
    counts0 = initial_counts(prior_mean, k, warm_start, initial_count_scenario)

    for item, n_obs in enumerate(counts0):
        for _ in range(int(n_obs)):
            y = w_true[item] + rng.normal(0.0, math.sqrt(obs_var[item]))
            mean0, cov0 = update_gaussian(mean0, cov0, item, y, obs_var[item])

    results: dict[str, list[float]] = {policy: [] for policy in POLICIES}
    budget_set = set(budgets)
    max_budget = max(budgets)

    for policy in POLICIES:
        policy_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
        mean = mean0.copy()
        cov = cov0.copy()
        counts = counts0.copy()
        checkpoints: list[float] = []

        for step in range(1, max_budget + 1):
            item = choose_item(policy, mean, cov, counts, obs_var, k, policy_rng)
            y = w_true[item] + policy_rng.normal(0.0, math.sqrt(obs_var[item]))
            counts[item] += 1
            mean, cov = update_gaussian(mean, cov, item, y, obs_var[item])
            if step in budget_set:
                chosen = np.argpartition(mean, -k)[-k:]
                checkpoints.append(oracle_sum - float(w_true[chosen].sum()))

        results[policy] = checkpoints

    return results


def summarize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    se = values.std(axis=0, ddof=1) / math.sqrt(values.shape[0])
    return mean, se


def main() -> None:
    args = parse_args()
    budget_multiples = parse_budget_multiples(args.budgets)
    budgets = [multiple * args.L for multiple in budget_multiples]
    gh_x, gh_w = np.polynomial.hermite.hermgauss(31)
    prior_cov = ar1_cov(args.L, args.rho)
    master_rng = np.random.default_rng(args.seed)

    records: dict[tuple[str, str, str], dict[str, list[list[float]]]] = {}
    cutoff_gaps: dict[tuple[str, str, str], list[float]] = {}
    raw_rows = []

    for density_name, spacing in DENSITY_SCENARIOS:
        for stock_name in INFO_STOCK_SCENARIOS:
            for info_name, info_multiplier in INFORMATION_SCALE_SCENARIOS:
                key = (density_name, stock_name, info_name)
                records[key] = {policy: [] for policy in POLICIES}
                cutoff_gaps[key] = []

                for world in range(args.worlds):
                    world_seed = int(master_rng.integers(0, 2**32 - 1))
                    world_rng = np.random.default_rng(world_seed)
                    prior_mean = make_prior_mean(world_rng, args.L, args.K, spacing)
                    cutoff_gap = local_cutoff_gap(prior_mean, args.K)
                    w_true = world_rng.multivariate_normal(prior_mean, prior_cov)
                    kappas = np.array(
                        [
                            info_multiplier * expected_kappa(float(mu), gh_x, gh_w)
                            for mu in prior_mean
                        ],
                        dtype=float,
                    )
                    world_results = run_world(
                        prior_mean=prior_mean,
                        prior_cov=prior_cov,
                        w_true=w_true,
                        kappas=kappas,
                        initial_count_scenario=stock_name,
                        budgets=budgets,
                        warm_start=args.warm_start,
                        k=args.K,
                        rng=world_rng,
                    )
                    cutoff_gaps[key].append(cutoff_gap)
                    for policy in POLICIES:
                        records[key][policy].append(world_results[policy])
                        for budget, budget_multiple, gap in zip(
                            budgets, budget_multiples, world_results[policy]
                        ):
                            raw_rows.append(
                                {
                                    "density_scenario": density_name,
                                    "information_stock": stock_name,
                                    "information_scale": info_name,
                                    "information_multiplier": info_multiplier,
                                    "world": world,
                                    "world_seed": world_seed,
                                    "mean_prior_cutoff_gap": cutoff_gap,
                                    "budget_multiple": budget_multiple,
                                    "budget": budget,
                                    "policy": policy,
                                    "gap": gap,
                                }
                            )

    rows = []
    for key, policy_records in records.items():
        density_name, stock_name, info_name = key
        info_multiplier = dict(INFORMATION_SCALE_SCENARIOS)[info_name]
        stacked = {
            policy: np.array(policy_records[policy], dtype=float)
            for policy in POLICIES
        }
        cutoff_mean, cutoff_se = summarize(stacked["cutoff_only"])
        combined_mean, combined_se = summarize(stacked["cutoff_x_uncertainty"])
        cutoff_gap_arr = np.array(cutoff_gaps[key], dtype=float)

        for idx, (budget, budget_multiple) in enumerate(zip(budgets, budget_multiples)):
            improvement = (
                stacked["cutoff_only"][:, idx]
                - stacked["cutoff_x_uncertainty"][:, idx]
            )
            cutoff_gap = cutoff_mean[idx]
            combined_gap = combined_mean[idx]
            rows.append(
                {
                    "density_scenario": density_name,
                    "information_stock": stock_name,
                    "information_scale": info_name,
                    "information_multiplier": info_multiplier,
                    "worlds": args.worlds,
                    "mean_prior_cutoff_gap": cutoff_gap_arr.mean(),
                    "budget_multiple": budget_multiple,
                    "budget": budget,
                    "cutoff_mean": cutoff_gap,
                    "cutoff_se": cutoff_se[idx],
                    "combined_mean": combined_gap,
                    "combined_se": combined_se[idx],
                    "combined_improvement_vs_cutoff_mean": improvement.mean(),
                    "combined_improvement_vs_cutoff_se": scalar_se(improvement),
                    "combined_win_rate_vs_cutoff": (
                        stacked["cutoff_x_uncertainty"][:, idx]
                        < stacked["cutoff_only"][:, idx]
                    ).mean(),
                    "combined_pct_improvement_vs_cutoff": 100.0
                    * improvement.mean()
                    / max(cutoff_gap, 1e-8),
                    "best_policy_by_mean": (
                        "cutoff_x_uncertainty" if combined_gap < cutoff_gap else "cutoff_only"
                    ),
                }
            )

    with Path(args.raw_output).open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "density_scenario",
                "information_stock",
                "information_scale",
                "information_multiplier",
                "world",
                "world_seed",
                "mean_prior_cutoff_gap",
                "budget_multiple",
                "budget",
                "policy",
                "gap",
            ],
        )
        writer.writeheader()
        writer.writerows(raw_rows)

    with Path(args.output).open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "density_scenario",
                "information_stock",
                "information_scale",
                "information_multiplier",
                "worlds",
                "mean_prior_cutoff_gap",
                "budget_multiple",
                "budget",
                "cutoff_mean",
                "cutoff_se",
                "combined_mean",
                "combined_se",
                "combined_improvement_vs_cutoff_mean",
                "combined_improvement_vs_cutoff_se",
                "combined_win_rate_vs_cutoff",
                "combined_pct_improvement_vs_cutoff",
                "best_policy_by_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
