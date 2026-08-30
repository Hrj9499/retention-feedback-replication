#!/usr/bin/env python3
"""Synthetic allocation-policy comparison for Section 7.2.

The simulation is intentionally manager-facing rather than algorithmic:
we begin from a common warm start, then compare how additional feedback
budgets are deployed under transparent rules. Additional feedback is
simulated through a Gaussian information-equivalent observation channel:
the signal variance for item i is 1/kappa_i(S), where kappa_i(S) is
computed from the ordered-probit Fisher information.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


THRESHOLDS = [-np.inf, -1.0, -0.2, 0.2, 1.0, np.inf]
POLICIES = (
    "even",
    "uncertainty_only",
    "cutoff_only",
    "cutoff_x_uncertainty",
    "local_marginal",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=60)
    parser.add_argument("--K", type=int, default=6)
    parser.add_argument("--rho", type=float, default=0.6)
    parser.add_argument("--S", type=int, default=5)
    parser.add_argument("--worlds", type=int, default=100)
    parser.add_argument("--warm-start", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260406)
    parser.add_argument(
        "--output",
        default="synthetic_allocation_summary.csv",
        help="CSV path for the summary table.",
    )
    parser.add_argument(
        "--raw-output",
        default="synthetic_allocation_raw.csv",
        help="CSV path for one row per prior/world/budget/policy.",
    )
    parser.add_argument(
        "--budget-multiples",
        default="10,30,100",
        help="Comma-separated budget checkpoints in multiples of L; 0 records the warm-start baseline.",
    )
    return parser.parse_args()


def phi(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def Phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def fisher_info(w: float) -> float:
    total = 0.0
    for lo, hi in zip(THRESHOLDS[:-1], THRESHOLDS[1:]):
        Phi_hi = 1.0 if np.isposinf(hi) else Phi(hi - w)
        Phi_lo = 0.0 if np.isneginf(lo) else Phi(lo - w)
        prob = max(Phi_hi - Phi_lo, 1e-12)
        d_hi = 0.0 if np.isposinf(hi) else phi(hi - w)
        d_lo = 0.0 if np.isneginf(lo) else phi(lo - w)
        deriv = d_lo - d_hi
        total += (deriv * deriv) / prob
    return total


def expected_kappa(mu: float, gh_x: np.ndarray, gh_w: np.ndarray) -> float:
    weighted = [
        weight * fisher_info(mu + math.sqrt(2.0) * x)
        for x, weight in zip(gh_x, gh_w)
    ]
    return float(sum(weighted) / math.sqrt(math.pi))


def ar1_cov(size: int, rho: float) -> np.ndarray:
    idx = np.arange(size)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def boundary_weight(means: np.ndarray, k: int, floor: float = 0.10) -> np.ndarray:
    sorted_desc = np.sort(means)[::-1]
    cutoff = sorted_desc[k - 1]
    local_gaps = []
    if k - 2 >= 0:
        local_gaps.append(abs(sorted_desc[k - 2] - cutoff))
    if k < len(sorted_desc):
        local_gaps.append(abs(cutoff - sorted_desc[k]))
    cross_section = np.std(means)
    scale = max(np.median(local_gaps) if local_gaps else 0.0, 0.10 * cross_section, 0.08)
    diff = np.abs(means - cutoff)
    base = np.exp(-((diff / scale) ** 2))
    return floor + (1.0 - floor) * base


def total_variance_drop(cov: np.ndarray, obs_var: np.ndarray) -> np.ndarray:
    diag = np.diag(cov)
    return np.sum(cov * cov, axis=0) / (diag + obs_var)


def local_marginal_score(
    mean: np.ndarray,
    cov: np.ndarray,
    kappas: np.ndarray,
    k: int,
) -> np.ndarray:
    """Gaussian-proxy local marginal score corresponding to Proposition 9.

    The infinitesimal covariance derivative for one more information unit on
    item j is kappa_j A e_j e_j' A. Weighting the affected variances by the
    current boundary score gives sum_i b_i kappa_j A_ij^2. Under independence
    this reduces to b_j kappa_j / (tau_j + kappa_j N_j)^2.
    """
    boundary = boundary_weight(mean, k, floor=0.0)
    spillover_value = (boundary[:, None] * (cov * cov)).sum(axis=0)
    return kappas * spillover_value


def update_gaussian(
    mean: np.ndarray, cov: np.ndarray, item: int, y: float, obs_var: float
) -> tuple[np.ndarray, np.ndarray]:
    denom = cov[item, item] + obs_var
    gain = cov[:, item] / denom
    innov = y - mean[item]
    new_mean = mean + gain * innov
    new_cov = cov - np.outer(gain, cov[item, :])
    new_cov = (new_cov + new_cov.T) / 2.0
    return new_mean, new_cov


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

    cutoff_score = boundary_weight(mean, k)
    uncertainty_score = total_variance_drop(cov, obs_var)

    if policy == "uncertainty_only":
        scores = uncertainty_score
    elif policy == "cutoff_only":
        # A one-dimensional cutoff rule should still acknowledge
        # diminishing returns from repeatedly asking about the same item.
        scores = cutoff_score / (1.0 + 0.10 * counts)
    elif policy == "cutoff_x_uncertainty":
        scores = (
            cutoff_score
            * (0.15 + 0.85 * uncertainty_score / uncertainty_score.max())
            / (1.0 + 0.05 * counts)
        )
    elif policy == "local_marginal":
        scores = local_marginal_score(mean, cov, kappas, k)
    else:
        raise ValueError(f"Unknown policy: {policy}")

    max_score = scores.max()
    best = np.flatnonzero(np.isclose(scores, max_score))
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

    results: dict[str, list[float]] = {policy: [] for policy in POLICIES}
    budget_set = set(budgets)
    max_budget = max(budgets)

    for policy in POLICIES:
        mean = mean0.copy()
        cov = cov0.copy()
        counts = counts0.copy()
        policy_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))

        checkpoints: list[float] = []
        if 0 in budget_set:
            chosen = np.argpartition(mean, -k)[-k:]
            checkpoints.append(oracle_sum - float(w_true[chosen].sum()))
        for step in range(1, max_budget + 1):
            item = choose_item(policy, mean, cov, counts, kappas, obs_var, k, policy_rng)
            y = w_true[item] + policy_rng.normal(0.0, math.sqrt(obs_var[item]))
            counts[item] += 1
            mean, cov = update_gaussian(mean, cov, item, y, obs_var[item])
            if step in budget_set:
                chosen = np.argpartition(mean, -k)[-k:]
                gap = oracle_sum - float(w_true[chosen].sum())
                checkpoints.append(gap)

        results[policy] = checkpoints

    return results


def summarize(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = arr.mean(axis=0)
    se = arr.std(axis=0, ddof=1) / math.sqrt(arr.shape[0])
    return mean, se


def scalar_se(values: np.ndarray) -> float:
    if len(values) <= 1:
        return float("nan")
    return float(values.std(ddof=1) / math.sqrt(len(values)))


def main() -> None:
    args = parse_args()
    budgets = [int(round(float(m) * args.L)) for m in args.budget_multiples.split(",")]
    gh_x, gh_w = np.polynomial.hermite.hermgauss(31)
    output_path = Path(args.output)
    raw_output_path = Path(args.raw_output)
    master_rng = np.random.default_rng(args.seed)

    priors = {
        "AR(1) $\\\\rho=0.6$": ar1_cov(args.L, args.rho),
        "Independent ($\\\\bSigma{=}I$)": np.eye(args.L),
    }

    records: dict[str, dict[str, list[list[float]]]] = {
        prior_name: {policy: [] for policy in POLICIES}
        for prior_name in priors
    }
    raw_rows = []

    for prior_name, prior_cov in priors.items():
        for world in range(args.worlds):
            world_seed = int(master_rng.integers(0, 2**32 - 1))
            world_rng = np.random.default_rng(world_seed)
            prior_mean = world_rng.normal(0.0, 1.0, size=args.L)
            w_true = world_rng.multivariate_normal(prior_mean, prior_cov)
            kappas = np.array(
                [expected_kappa(float(mu), gh_x, gh_w) for mu in prior_mean], dtype=float
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
            for policy in POLICIES:
                records[prior_name][policy].append(world_results[policy])
                for budget, gap in zip(budgets, world_results[policy]):
                    raw_rows.append(
                        {
                            "prior": prior_name,
                            "world": world,
                            "world_seed": world_seed,
                            "budget": budget,
                            "policy": policy,
                            "gap": gap,
                        }
                    )

    rows = []
    for prior_name in priors:
        stacked = {
            policy: np.array(records[prior_name][policy], dtype=float)
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

        benchmark_stack = np.minimum.reduce([stacked[policy] for policy in POLICIES])
        benchmark_mean, benchmark_se = summarize(benchmark_stack)

        for idx, budget in enumerate(budgets):
            even_gap = means["even"][idx]
            cutoff_gap = means["cutoff_only"][idx]
            combined_gap = means["cutoff_x_uncertainty"][idx]
            local_marginal_gap = means["local_marginal"][idx]
            cutoff_pct_reduction = 100.0 * (even_gap - cutoff_gap) / max(even_gap, 1e-8)
            combined_pct_reduction = (
                100.0 * (even_gap - combined_gap) / max(even_gap, 1e-8)
            )
            local_marginal_pct_reduction = (
                100.0 * (even_gap - local_marginal_gap) / max(even_gap, 1e-8)
            )
            cutoff_improvement = stacked["even"][:, idx] - stacked["cutoff_only"][:, idx]
            combined_improvement = (
                stacked["even"][:, idx] - stacked["cutoff_x_uncertainty"][:, idx]
            )
            local_marginal_improvement = (
                stacked["even"][:, idx] - stacked["local_marginal"][:, idx]
            )
            policy_means = {policy: means[policy][idx] for policy in POLICIES}
            best_policy = min(policy_means, key=policy_means.get)
            rows.append(
                {
                    "prior": prior_name,
                    "worlds": args.worlds,
                    "budget": budget,
                    "even_mean": means["even"][idx],
                    "even_se": ses["even"][idx],
                    "even_win_rate": win_rates["even"][idx],
                    "uncertainty_mean": means["uncertainty_only"][idx],
                    "uncertainty_se": ses["uncertainty_only"][idx],
                    "uncertainty_win_rate": win_rates["uncertainty_only"][idx],
                    "cutoff_mean": means["cutoff_only"][idx],
                    "cutoff_se": ses["cutoff_only"][idx],
                    "cutoff_win_rate": win_rates["cutoff_only"][idx],
                    "combined_mean": means["cutoff_x_uncertainty"][idx],
                    "combined_se": ses["cutoff_x_uncertainty"][idx],
                    "combined_win_rate": win_rates["cutoff_x_uncertainty"][idx],
                    "local_marginal_mean": means["local_marginal"][idx],
                    "local_marginal_se": ses["local_marginal"][idx],
                    "local_marginal_win_rate": win_rates["local_marginal"][idx],
                    "benchmark_mean": benchmark_mean[idx],
                    "benchmark_se": benchmark_se[idx],
                    "cutoff_pct_gap_reduction": cutoff_pct_reduction,
                    "combined_pct_gap_reduction": combined_pct_reduction,
                    "local_marginal_pct_gap_reduction": local_marginal_pct_reduction,
                    "pct_gap_reduction": combined_pct_reduction,
                    "cutoff_improvement_vs_even_mean": cutoff_improvement.mean(),
                    "cutoff_improvement_vs_even_se": scalar_se(cutoff_improvement),
                    "combined_improvement_vs_even_mean": combined_improvement.mean(),
                    "combined_improvement_vs_even_se": scalar_se(combined_improvement),
                    "local_marginal_improvement_vs_even_mean": local_marginal_improvement.mean(),
                    "local_marginal_improvement_vs_even_se": scalar_se(local_marginal_improvement),
                    "best_policy_by_mean": best_policy,
                }
            )

    with raw_output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "prior",
                "world",
                "world_seed",
                "budget",
                "policy",
                "gap",
            ],
        )
        writer.writeheader()
        writer.writerows(raw_rows)

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "prior",
                "worlds",
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
                "local_marginal_mean",
                "local_marginal_se",
                "local_marginal_win_rate",
                "benchmark_mean",
                "benchmark_se",
                "cutoff_pct_gap_reduction",
                "combined_pct_gap_reduction",
                "local_marginal_pct_gap_reduction",
                "pct_gap_reduction",
                "cutoff_improvement_vs_even_mean",
                "cutoff_improvement_vs_even_se",
                "combined_improvement_vs_even_mean",
                "combined_improvement_vs_even_se",
                "local_marginal_improvement_vs_even_mean",
                "local_marginal_improvement_vs_even_se",
                "best_policy_by_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
