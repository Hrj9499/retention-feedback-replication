"""Robustness of cutoff targeting under informative missingness.

In practice, customers with strong reactions answer more often. This
benchmark makes response arrival depend on the realized signal: a
prompted customer responds with probability
pi(y) = 0.2 + 0.6 * sigmoid(2 * (|y| - 1)), so extreme experiences are
about four times as likely to be recorded as neutral ones. Prompts
consume budget whether or not an answer arrives, and the platform
updates naively on the answers it sees, which biases posterior means
toward extremes. The question is whether cutoff targeting survives
this selection.

Policies: even, guarded cutoff-only, combined cutoff x uncertainty,
each under full response and under informative missingness. World
protocol matches synthetic_allocation_benchmarks.py (AR(1) prior).
Outputs informative_missingness_summary.csv.
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
    total_variance_drop,
    update_gaussian,
)


def respond_prob(y):
    return 0.2 + 0.6 / (1.0 + math.exp(-2.0 * (abs(y) - 1.0)))


def choose(policy, mean, cov, counts, obs_var, k, rng):
    if policy == "even":
        eligible = np.flatnonzero(counts == counts.min())
        return int(rng.choice(eligible))
    cutoff_score = boundary_weight(mean, k)
    if policy == "guarded_cutoff":
        scores = cutoff_score / (1.0 + 0.10 * counts)
    elif policy == "combined":
        unc = total_variance_drop(cov, obs_var)
        scores = cutoff_score * (0.15 + 0.85 * unc / unc.max()) / (1.0 + 0.05 * counts)
    else:
        raise ValueError(policy)
    best = np.flatnonzero(np.isclose(scores, scores.max()))
    return int(rng.choice(best))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, default=100)
    parser.add_argument("--L", type=int, default=60)
    parser.add_argument("--K", type=int, default=6)
    parser.add_argument("--rho", type=float, default=0.6)
    parser.add_argument("--warm-start", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260406)
    parser.add_argument(
        "--out", type=Path, default=Path("informative_missingness_summary.csv")
    )
    args = parser.parse_args()

    L, K = args.L, args.K
    budgets = [10 * L, 30 * L]
    gh_x, gh_w = np.polynomial.hermite.hermgauss(31)
    master_rng = np.random.default_rng(args.seed)
    prior_cov = ar1_cov(L, args.rho)
    policies = ("even", "guarded_cutoff", "combined")
    channels = ("full response", "informative missingness")

    records = {(pol, ch): [] for pol in policies for ch in channels}
    for _ in range(args.worlds):
        world_seed = int(master_rng.integers(0, 2**32 - 1))
        world_rng = np.random.default_rng(world_seed)
        prior_mean = world_rng.normal(0.0, 1.0, size=L)
        w_true = world_rng.multivariate_normal(prior_mean, prior_cov)
        kappas = np.array(
            [expected_kappa(float(mu), gh_x, gh_w) for mu in prior_mean]
        )
        obs_var = 1.0 / np.clip(kappas, 1e-8, None)
        oracle = float(np.sort(w_true)[-K:].sum())

        mean0 = prior_mean.copy()
        cov0 = prior_cov.copy()
        counts0 = np.zeros(L)
        for item in range(L):
            for _ in range(args.warm_start):
                y = w_true[item] + world_rng.normal(0.0, math.sqrt(obs_var[item]))
                counts0[item] += 1
                mean0, cov0 = update_gaussian(mean0, cov0, item, y, obs_var[item])

        for pol in policies:
            for ch in channels:
                mean, cov, counts = mean0.copy(), cov0.copy(), counts0.copy()
                prng = np.random.default_rng(int(world_rng.integers(0, 2**32 - 1)))
                cps = []
                budget_set = set(budgets)
                for step in range(1, max(budgets) + 1):
                    j = choose(pol, mean, cov, counts, obs_var, K, prng)
                    counts[j] += 1  # the prompt consumes budget either way
                    y = w_true[j] + prng.normal(0.0, math.sqrt(obs_var[j]))
                    arrived = (
                        ch == "full response"
                        or prng.random() < respond_prob(y)
                    )
                    if arrived:
                        mean, cov = update_gaussian(mean, cov, j, y, obs_var[j])
                    if step in budget_set:
                        chosen = np.argpartition(mean, -K)[-K:]
                        cps.append(oracle - float(w_true[chosen].sum()))
                records[(pol, ch)].append(cps)

    rows = []
    for (pol, ch), worlds in records.items():
        arr = np.array(worlds)
        for b_idx, budget in enumerate(budgets):
            rows.append(
                {
                    "policy": pol,
                    "channel": ch,
                    "budget": budget,
                    "mean_gap": round(float(arr[:, b_idx].mean()), 4),
                    "se": round(
                        float(arr[:, b_idx].std(ddof=1) / math.sqrt(arr.shape[0])), 4
                    ),
                    "worlds": arr.shape[0],
                }
            )

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        f.write(
            f"# L={L} K={K} warm_start={args.warm_start} budgets={budgets} "
            f"seed={args.seed}; AR(1) rho={args.rho}; response probability "
            "0.2 + 0.6*sigmoid(2(|y|-1)); prompts consume budget regardless "
            "of arrival; naive updating on arrived answers.\n"
        )
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
