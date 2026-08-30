#!/usr/bin/env python3
"""Correlation-aware allocation diagnostics.

These benchmarks ask whether using the full posterior covariance to value
spillovers can change allocation performance relative to a cutoff rule that
sees only diagonal residual variances. The observation channel is the same
Gaussian information-equivalent proxy used in the main synthetic allocation
benchmark.
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
    summarize,
    update_gaussian,
)


POLICIES = ("even", "cutoff_diag_uncertainty", "cutoff_full_spillover")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=40)
    parser.add_argument("--K", type=int, default=6)
    parser.add_argument("--rho", type=float, default=0.7)
    parser.add_argument("--worlds", type=int, default=50)
    parser.add_argument("--warm-start", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument(
        "--output",
        default="correlation_spillover_summary.csv",
        help="CSV path for the summary table.",
    )
    parser.add_argument(
        "--raw-output",
        default="correlation_spillover_raw.csv",
        help="CSV path for one row per world, budget, and policy.",
    )
    return parser.parse_args()


def diagonal_cutoff_score(
    mean: np.ndarray,
    cov: np.ndarray,
    obs_var: np.ndarray,
    counts: np.ndarray,
    k: int,
) -> np.ndarray:
    boundary = boundary_weight(mean, k)
    diag = np.diag(cov)
    direct_drop = (diag * diag) / (diag + obs_var)
    return boundary * direct_drop / (1.0 + 0.05 * counts)


def full_spillover_score(
    mean: np.ndarray,
    cov: np.ndarray,
    obs_var: np.ndarray,
    counts: np.ndarray,
    k: int,
) -> np.ndarray:
    boundary = boundary_weight(mean, k)
    # Observing item j reduces covariance by cov[:,j] cov[j,:] / denom_j.
    # Value that deterministic covariance reduction across all products that
    # remain relevant to the cutoff, rather than only on item j's diagonal.
    denom = np.diag(cov) + obs_var
    spillover_drop = (boundary[:, None] * (cov * cov)).sum(axis=0) / denom
    return spillover_drop / (1.0 + 0.05 * counts)


def choose_item(
    policy: str,
    mean: np.ndarray,
    cov: np.ndarray,
    obs_var: np.ndarray,
    counts: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> int:
    if policy == "even":
        eligible = np.flatnonzero(counts == counts.min())
        return int(rng.choice(eligible))
    if policy == "cutoff_diag_uncertainty":
        scores = diagonal_cutoff_score(mean, cov, obs_var, counts, k)
    elif policy == "cutoff_full_spillover":
        scores = full_spillover_score(mean, cov, obs_var, counts, k)
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
    warm_start: int,
    k: int,
    rng: np.random.Generator,
) -> dict[str, list[float]]:
    obs_var = 1.0 / np.clip(kappas, 1e-8, None)
    oracle_sum = float(np.sort(w_true)[-k:].sum())

    mean0 = prior_mean.copy()
    cov0 = prior_cov.copy()
    counts0 = np.zeros(len(prior_mean), dtype=int)
    for item in range(len(prior_mean)):
        for _ in range(warm_start):
            y = w_true[item] + rng.normal(0.0, math.sqrt(obs_var[item]))
            counts0[item] += 1
            mean0, cov0 = update_gaussian(mean0, cov0, item, y, obs_var[item])

    budget_set = set(budgets)
    max_budget = max(budgets)
    results: dict[str, list[float]] = {policy: [] for policy in POLICIES}

    for policy in POLICIES:
        mean = mean0.copy()
        cov = cov0.copy()
        counts = counts0.copy()
        policy_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
        checkpoints: list[float] = []
        for step in range(1, max_budget + 1):
            item = choose_item(policy, mean, cov, obs_var, counts, k, policy_rng)
            y = w_true[item] + policy_rng.normal(0.0, math.sqrt(obs_var[item]))
            counts[item] += 1
            mean, cov = update_gaussian(mean, cov, item, y, obs_var[item])
            if step in budget_set:
                chosen = np.argpartition(mean, -k)[-k:]
                checkpoints.append(oracle_sum - float(w_true[chosen].sum()))
        results[policy] = checkpoints
    return results


def clustered_cov(size: int, block_size: int = 8) -> np.ndarray:
    """Block covariance with strongly correlated substitute clusters."""
    cov = np.full((size, size), 0.01)
    np.fill_diagonal(cov, 1.0)
    for start in range(0, min(24, size), block_size):
        end = min(size, start + block_size)
        cov[start:end, start:end] = 0.88
        np.fill_diagonal(cov[start:end, start:end], 1.0)
    if size > 24:
        cov[24:, 24:] = 0.05
        np.fill_diagonal(cov[24:, 24:], 1.0)
    min_eig = float(np.linalg.eigvalsh(cov).min())
    if min_eig <= 1e-8:
        cov += np.eye(size) * (1e-6 - min_eig)
    return cov


def clustered_prior_mean(rng: np.random.Generator, size: int) -> np.ndarray:
    """Create clusters with off-boundary proxy items and boundary substitutes."""
    mean = np.linspace(-1.0, -2.0, size)
    if size >= 28:
        mean[24:28] = np.linspace(1.40, 0.75, 4)
    for block in range(3):
        start = block * 8
        if start + 8 > size:
            break
        mean[start] = -0.90 + rng.normal(0.0, 0.01)  # off-boundary proxy item
        mean[start + 1 : start + 7] = (
            np.linspace(0.18, -0.12, 6) + rng.normal(0.0, 0.01, 6)
        )
        mean[start + 7] = -0.50 + rng.normal(0.0, 0.01)
    return mean


def build_world(
    scenario: str,
    args: argparse.Namespace,
    rng: np.random.Generator,
    gh_x: np.ndarray,
    gh_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int], int, str]:
    if scenario == "AR(1) random catalog":
        prior_cov = ar1_cov(args.L, args.rho)
        prior_mean = rng.normal(0.0, 1.0, size=args.L)
        w_true = rng.multivariate_normal(prior_mean, prior_cov)
        kappas = np.array(
            [expected_kappa(float(mu), gh_x, gh_w) for mu in prior_mean],
            dtype=float,
        )
        return (
            prior_mean,
            prior_cov,
            w_true,
            kappas,
            [10 * args.L, 30 * args.L],
            args.warm_start,
            f"rho={args.rho}",
        )

    if scenario == "Clustered boundary substitutes":
        prior_cov = clustered_cov(args.L)
        prior_mean = clustered_prior_mean(rng, args.L)
        w_true = rng.multivariate_normal(prior_mean, prior_cov)
        kappas = np.array(
            [expected_kappa(float(mu), gh_x, gh_w) for mu in prior_mean],
            dtype=float,
        )
        # In each correlated cluster, item 0 is an off-boundary proxy whose
        # feedback is relatively informative about a group of borderline
        # substitutes; the boundary substitutes are noisier to ask about
        # directly. This makes the value of covariance spillovers visible.
        for block in range(3):
            start = block * 8
            if start + 7 > args.L:
                break
            kappas[start] *= 12.0
            kappas[start + 1 : start + 7] *= 0.10
        return (
            prior_mean,
            prior_cov,
            w_true,
            kappas,
            [args.L, 3 * args.L, 6 * args.L],
            2,
            "block rho=0.88",
        )

    raise ValueError(f"Unknown scenario: {scenario}")


def main() -> None:
    args = parse_args()
    gh_x, gh_w = np.polynomial.hermite.hermgauss(31)
    master_rng = np.random.default_rng(args.seed)
    scenarios = ("AR(1) random catalog", "Clustered boundary substitutes")

    records = {scenario: {policy: [] for policy in POLICIES} for scenario in scenarios}
    budget_records: dict[str, list[int]] = {scenario: [] for scenario in scenarios}
    detail_records: dict[str, str] = {}
    raw_rows = []

    for scenario in scenarios:
        for world in range(args.worlds):
            world_seed = int(master_rng.integers(0, 2**32 - 1))
            world_rng = np.random.default_rng(world_seed)
            prior_mean, prior_cov, w_true, kappas, budgets, warm_start, detail = build_world(
                scenario, args, world_rng, gh_x, gh_w
            )
            if not budget_records[scenario]:
                budget_records[scenario] = budgets
                detail_records[scenario] = detail
            world_results = run_world(
                prior_mean=prior_mean,
                prior_cov=prior_cov,
                w_true=w_true,
                kappas=kappas,
                budgets=budgets,
                warm_start=warm_start,
                k=args.K,
                rng=world_rng,
            )
            for policy in POLICIES:
                records[scenario][policy].append(world_results[policy])
                for budget, gap in zip(budgets, world_results[policy]):
                    raw_rows.append(
                        {
                            "scenario": scenario,
                            "world": world,
                            "world_seed": world_seed,
                            "scenario_detail": detail,
                            "budget": budget,
                            "policy": policy,
                            "gap": gap,
                        }
                )

    rows = []
    for scenario in scenarios:
        stacked = {
            policy: np.array(records[scenario][policy], dtype=float)
            for policy in POLICIES
        }
        means = {}
        ses = {}
        for policy, values in stacked.items():
            means[policy], ses[policy] = summarize(values)

        for idx, budget in enumerate(budget_records[scenario]):
            diag_improvement = (
                stacked["even"][:, idx] - stacked["cutoff_diag_uncertainty"][:, idx]
            )
            full_improvement = (
                stacked["even"][:, idx] - stacked["cutoff_full_spillover"][:, idx]
            )
            full_vs_diag = (
                stacked["cutoff_diag_uncertainty"][:, idx]
                - stacked["cutoff_full_spillover"][:, idx]
            )
            rows.append(
                {
                    "scenario": scenario,
                    "scenario_detail": detail_records[scenario],
                    "worlds": args.worlds,
                    "budget": budget,
                    "even_mean": means["even"][idx],
                    "even_se": ses["even"][idx],
                    "diag_cutoff_mean": means["cutoff_diag_uncertainty"][idx],
                    "diag_cutoff_se": ses["cutoff_diag_uncertainty"][idx],
                    "full_spillover_mean": means["cutoff_full_spillover"][idx],
                    "full_spillover_se": ses["cutoff_full_spillover"][idx],
                    "diag_improvement_vs_even_mean": diag_improvement.mean(),
                    "diag_improvement_vs_even_se": scalar_se(diag_improvement),
                    "full_improvement_vs_even_mean": full_improvement.mean(),
                    "full_improvement_vs_even_se": scalar_se(full_improvement),
                    "full_improvement_vs_diag_mean": full_vs_diag.mean(),
                    "full_improvement_vs_diag_se": scalar_se(full_vs_diag),
                    "full_win_rate_vs_diag": float((full_vs_diag > 0.0).mean()),
                }
            )

    with Path(args.raw_output).open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario",
                "world",
                "world_seed",
                "scenario_detail",
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
                "scenario",
                "scenario_detail",
                "worlds",
                "budget",
                "even_mean",
                "even_se",
                "diag_cutoff_mean",
                "diag_cutoff_se",
                "full_spillover_mean",
                "full_spillover_se",
                "diag_improvement_vs_even_mean",
                "diag_improvement_vs_even_se",
                "full_improvement_vs_even_mean",
                "full_improvement_vs_even_se",
                "full_improvement_vs_diag_mean",
                "full_improvement_vs_diag_se",
                "full_win_rate_vs_diag",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
