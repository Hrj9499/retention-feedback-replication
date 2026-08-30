"""OCBA-m comparator on the main allocation protocol.

Adds the ranking-and-selection benchmark the paper cites as its closest
operations neighbor: a sequential implementation of optimal computing
budget allocation for selecting an optimal subset (Chen, He, Fu, and
Lee 2008). At each step the rule computes the OCBA-m target shares from
current posterior means, alpha_i proportional to (sigma_i / delta_i)^2
with delta_i the distance to the midpoint between the K-th and
(K+1)-th ranked means and sigma_i the observation standard deviation,
and assigns the next answer to the item furthest below its target
share. World generation matches synthetic_allocation_benchmarks.py.

Outputs ocba_summary.csv.
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
    update_gaussian,
)


def ocba_m_choice(mean, counts, obs_sd, k):
    order = np.argsort(mean)[::-1]
    c = 0.5 * (mean[order[k - 1]] + mean[order[k]])
    delta = np.abs(mean - c)
    spread = max(float(np.std(mean)), 1e-6)
    delta = np.maximum(delta, 0.05 * spread)
    target = (obs_sd / delta) ** 2
    target = target / target.sum()
    total = counts.sum() + 1.0
    deficit = target * total - counts
    return int(np.argmax(deficit))


def run_world(prior_mean, prior_cov, w_true, kappas, budgets, warm, k, rng):
    obs_var = 1.0 / np.clip(kappas, 1e-8, None)
    obs_sd = np.sqrt(obs_var)
    oracle = float(np.sort(w_true)[-k:].sum())
    L = len(prior_mean)

    mean0 = prior_mean.copy()
    cov0 = prior_cov.copy()
    counts0 = np.zeros(L)
    for item in range(L):
        for _ in range(warm):
            y = w_true[item] + rng.normal(0.0, math.sqrt(obs_var[item]))
            counts0[item] += 1
            mean0, cov0 = update_gaussian(mean0, cov0, item, y, obs_var[item])

    out = {}
    budget_set = set(budgets)
    for policy in ("guarded_cutoff", "ocba_m"):
        mean, cov, counts = mean0.copy(), cov0.copy(), counts0.copy()
        prng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
        checkpoints = []
        for step in range(1, max(budgets) + 1):
            if policy == "guarded_cutoff":
                scores = boundary_weight(mean, k) / (1.0 + 0.10 * counts)
                best = np.flatnonzero(np.isclose(scores, scores.max()))
                j = int(prng.choice(best))
            else:
                j = ocba_m_choice(mean, counts, obs_sd, k)
            y = w_true[j] + prng.normal(0.0, math.sqrt(obs_var[j]))
            counts[j] += 1
            mean, cov = update_gaussian(mean, cov, j, y, obs_var[j])
            if step in budget_set:
                chosen = np.argpartition(mean, -k)[-k:]
                checkpoints.append(oracle - float(w_true[chosen].sum()))
        out[policy] = checkpoints
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, default=100)
    parser.add_argument("--L", type=int, default=60)
    parser.add_argument("--K", type=int, default=6)
    parser.add_argument("--rho", type=float, default=0.6)
    parser.add_argument("--warm-start", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260406)
    parser.add_argument("--out", type=Path, default=Path("ocba_summary.csv"))
    args = parser.parse_args()

    budgets = [10 * args.L, 30 * args.L, 100 * args.L]
    gh_x, gh_w = np.polynomial.hermite.hermgauss(31)
    master_rng = np.random.default_rng(args.seed)
    priors = {
        "AR(1) rho=0.6": ar1_cov(args.L, args.rho),
        "Independent": np.eye(args.L),
    }

    records = {p: {pol: [] for pol in ("guarded_cutoff", "ocba_m")} for p in priors}
    for prior_name, prior_cov in priors.items():
        for _ in range(args.worlds):
            world_seed = int(master_rng.integers(0, 2**32 - 1))
            world_rng = np.random.default_rng(world_seed)
            prior_mean = world_rng.normal(0.0, 1.0, size=args.L)
            w_true = world_rng.multivariate_normal(prior_mean, prior_cov)
            kappas = np.array(
                [expected_kappa(float(mu), gh_x, gh_w) for mu in prior_mean]
            )
            res = run_world(
                prior_mean, prior_cov, w_true, kappas, budgets,
                args.warm_start, args.K, world_rng,
            )
            for pol, cps in res.items():
                records[prior_name][pol].append(cps)

    rows = []
    for prior_name in priors:
        for pol in ("guarded_cutoff", "ocba_m"):
            arr = np.array(records[prior_name][pol])
            for b_idx, budget in enumerate(budgets):
                rows.append(
                    {
                        "prior": prior_name,
                        "budget": budget,
                        "policy": pol,
                        "mean_gap": round(float(arr[:, b_idx].mean()), 4),
                        "se": round(
                            float(arr[:, b_idx].std(ddof=1) / math.sqrt(arr.shape[0])),
                            4,
                        ),
                        "worlds": arr.shape[0],
                    }
                )

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        f.write(
            f"# L={args.L} K={args.K} warm_start={args.warm_start} "
            f"budgets={budgets} seed={args.seed}; sequential OCBA-m with "
            "delta floored at 5% of the cross-sectional sd; world protocol "
            "matches synthetic_allocation_benchmarks.py.\n"
        )
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
