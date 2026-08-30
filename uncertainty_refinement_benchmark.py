#!/usr/bin/env python3
"""Synthetic stress test where uncertainty refines boundary targeting.

The main allocation benchmark uses a common warm start, where cutoff-only
targeting captures most of the value. This diagnostic constructs the opposite
case: many products sit near the shipment cutoff, but their initial information
stocks differ sharply. Some boundary products are already measured precisely,
while nearby rivals remain poorly measured. In that regime, a boundary-only
rule can keep asking about already-resolved borderline products; multiplying
boundary relevance by marginal uncertainty redirects questions to the products
where one more response still has value.
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


POLICIES = ("even", "uncertainty_only", "boundary_only", "cutoff_x_uncertainty")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=60)
    parser.add_argument("--K", type=int, default=6)
    parser.add_argument("--rho", type=float, default=0.6)
    parser.add_argument("--worlds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument(
        "--output",
        default="uncertainty_refinement_summary.csv",
        help="CSV path for the summary table.",
    )
    parser.add_argument(
        "--raw-output",
        default="uncertainty_refinement_raw.csv",
        help="CSV path for one row per world, budget, and policy.",
    )
    return parser.parse_args()


def make_prior_mean(rng: np.random.Generator, size: int, k: int) -> np.ndarray:
    """Build a catalog with a deliberately crowded top-K boundary."""
    values: list[float] = []
    values.extend(np.linspace(1.40, 0.75, max(k - 3, 0)))

    boundary_count = 14
    boundary_values = np.linspace(0.18, -0.18, boundary_count)
    boundary_values += rng.normal(0.0, 0.01, boundary_count)
    values.extend(boundary_values.tolist())

    remaining = size - len(values)
    if remaining > 0:
        values.extend(np.linspace(-0.70, -2.20, remaining).tolist())

    ordered = np.array(values[:size], dtype=float)
    return ordered[rng.permutation(size)]


def initial_counts(prior_mean: np.ndarray, k: int) -> np.ndarray:
    """Give equally plausible boundary products very unequal information stocks."""
    order = np.argsort(-prior_mean)
    counts = np.full(len(prior_mean), 5, dtype=int)

    obvious_top = order[: max(0, k - 3)]
    counts[obvious_top] = 30

    boundary_band = order[max(0, k - 3) : min(len(prior_mean), k + 9)]
    for offset, item in enumerate(boundary_band):
        counts[item] = 80 if offset % 2 == 0 else 1

    return counts


def choose_item(
    policy: str,
    mean: np.ndarray,
    cov: np.ndarray,
    counts: np.ndarray,
    kappas: np.ndarray,
    obs_var: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> int:
    if policy == "even":
        eligible = np.flatnonzero(counts == counts.min())
        return int(rng.choice(eligible))

    boundary_score = boundary_weight(mean, k, floor=0.02)
    uncertainty_score = total_variance_drop(cov, obs_var)

    if policy == "uncertainty_only":
        scores = uncertainty_score
    elif policy == "boundary_only":
        scores = boundary_score
    elif policy == "cutoff_x_uncertainty":
        scores = boundary_score * uncertainty_score
    else:
        raise ValueError(f"Unknown policy: {policy}")

    best = np.flatnonzero(np.isclose(scores, scores.max()))
    return int(rng.choice(best))


def run_world(
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
    w_true: np.ndarray,
    kappas: np.ndarray,
    budgets: list[int],
    k: int,
    rng: np.random.Generator,
) -> dict[str, list[float]]:
    obs_var = 1.0 / np.clip(kappas, 1e-8, None)
    oracle_sum = float(np.sort(w_true)[-k:].sum())

    mean0 = prior_mean.copy()
    cov0 = prior_cov.copy()
    counts0 = initial_counts(prior_mean, k)

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
            item = choose_item(policy, mean, cov, counts, kappas, obs_var, k, policy_rng)
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
    budgets = [args.L, 3 * args.L, 5 * args.L]
    master_rng = np.random.default_rng(args.seed)
    gh_x, gh_w = np.polynomial.hermite.hermgauss(31)
    prior_cov = ar1_cov(args.L, args.rho)

    records: dict[str, list[list[float]]] = {policy: [] for policy in POLICIES}
    raw_rows = []

    for world in range(args.worlds):
        world_seed = int(master_rng.integers(0, 2**32 - 1))
        world_rng = np.random.default_rng(world_seed)
        prior_mean = make_prior_mean(world_rng, args.L, args.K)
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
            k=args.K,
            rng=world_rng,
        )
        for policy in POLICIES:
            records[policy].append(world_results[policy])
            for budget, gap in zip(budgets, world_results[policy]):
                raw_rows.append(
                    {
                        "world": world,
                        "world_seed": world_seed,
                        "budget": budget,
                        "policy": policy,
                        "gap": gap,
                    }
                )

    stacked = {policy: np.array(records[policy], dtype=float) for policy in POLICIES}
    means = {}
    ses = {}
    for policy, values in stacked.items():
        means[policy], ses[policy] = summarize(values)

    rows = []
    for idx, budget in enumerate(budgets):
        improvement_vs_boundary = (
            stacked["boundary_only"][:, idx] - stacked["cutoff_x_uncertainty"][:, idx]
        )
        win_vs_boundary = (
            stacked["cutoff_x_uncertainty"][:, idx] < stacked["boundary_only"][:, idx]
        ).mean()
        rows.append(
            {
                "scenario": "heterogeneous boundary information",
                "worlds": args.worlds,
                "rho": args.rho,
                "budget": budget,
                "even_mean": means["even"][idx],
                "even_se": ses["even"][idx],
                "uncertainty_mean": means["uncertainty_only"][idx],
                "uncertainty_se": ses["uncertainty_only"][idx],
                "boundary_mean": means["boundary_only"][idx],
                "boundary_se": ses["boundary_only"][idx],
                "combined_mean": means["cutoff_x_uncertainty"][idx],
                "combined_se": ses["cutoff_x_uncertainty"][idx],
                "combined_improvement_vs_boundary_mean": improvement_vs_boundary.mean(),
                "combined_improvement_vs_boundary_se": scalar_se(improvement_vs_boundary),
                "combined_win_rate_vs_boundary": win_vs_boundary,
            }
        )

    with Path(args.raw_output).open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["world", "world_seed", "budget", "policy", "gap"],
        )
        writer.writeheader()
        writer.writerows(raw_rows)

    with Path(args.output).open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario",
                "worlds",
                "rho",
                "budget",
                "even_mean",
                "even_se",
                "uncertainty_mean",
                "uncertainty_se",
                "boundary_mean",
                "boundary_se",
                "combined_mean",
                "combined_se",
                "combined_improvement_vs_boundary_mean",
                "combined_improvement_vs_boundary_se",
                "combined_win_rate_vs_boundary",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
