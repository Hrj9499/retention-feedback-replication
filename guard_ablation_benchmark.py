"""Ablation of the cutoff-only rule's implementation devices.

The published cutoff-only policy is not pure proximity. It uses a
Gaussian proximity kernel with a floor of 0.10 and divides scores by
(1 + 0.10 * counts), a response-count damping that acts as an
estimate-free proxy for the remaining-learnability factor in
Proposition 4. This ablation quantifies what each device contributes by
comparing, under the same worlds and budgets as the main allocation
experiment:

- cutoff_guarded: the published rule (kernel floor + count damping)
- cutoff_no_damping: kernel floor kept, count damping removed
- cutoff_pure: no floor, no damping (raw proximity kernel only)

Outputs guard_ablation_summary.csv.
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

ABLATION_POLICIES = ("cutoff_guarded", "cutoff_no_damping", "cutoff_pure")


def choose_item_ablation(
    policy: str,
    mean: np.ndarray,
    counts: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> int:
    if policy == "cutoff_guarded":
        scores = boundary_weight(mean, k) / (1.0 + 0.10 * counts)
    elif policy == "cutoff_no_damping":
        scores = boundary_weight(mean, k)
    elif policy == "cutoff_pure":
        scores = boundary_weight(mean, k, floor=0.0)
    else:
        raise ValueError(policy)
    best = np.flatnonzero(np.isclose(scores, scores.max()))
    return int(rng.choice(best))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, default=100)
    parser.add_argument("--L", type=int, default=60)
    parser.add_argument("--K", type=int, default=6)
    parser.add_argument("--rho", type=float, default=0.6)
    parser.add_argument("--warm-start", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260406)
    parser.add_argument("--out", type=Path, default=Path("guard_ablation_summary.csv"))
    args = parser.parse_args()

    budgets = [10 * args.L, 30 * args.L, 100 * args.L]
    gh_x, gh_w = np.polynomial.hermite.hermgauss(31)
    master_rng = np.random.default_rng(args.seed)

    priors = {
        "AR(1) rho=0.6": ar1_cov(args.L, args.rho),
        "Independent": np.eye(args.L),
    }

    records = {
        prior: {p: [] for p in ABLATION_POLICIES} for prior in priors
    }
    for prior_name, prior_cov in priors.items():
        for _ in range(args.worlds):
            world_seed = int(master_rng.integers(0, 2**32 - 1))
            world_rng = np.random.default_rng(world_seed)
            prior_mean = world_rng.normal(0.0, 1.0, size=args.L)
            w_true = world_rng.multivariate_normal(prior_mean, prior_cov)
            kappas = np.array(
                [expected_kappa(float(mu), gh_x, gh_w) for mu in prior_mean]
            )
            obs_var = 1.0 / np.clip(kappas, 1e-8, None)
            oracle_sum = float(np.sort(w_true)[-args.K:].sum())

            mean0 = prior_mean.copy()
            cov0 = prior_cov.copy()
            counts0 = np.zeros(args.L, dtype=int)
            for item in range(args.L):
                for _ in range(args.warm_start):
                    y = w_true[item] + world_rng.normal(
                        0.0, math.sqrt(obs_var[item])
                    )
                    counts0[item] += 1
                    mean0, cov0 = update_gaussian(
                        mean0, cov0, item, y, obs_var[item]
                    )

            budget_set = set(budgets)
            for policy in ABLATION_POLICIES:
                mean = mean0.copy()
                cov = cov0.copy()
                counts = counts0.copy()
                policy_rng = np.random.default_rng(
                    int(world_rng.integers(0, 2**32 - 1))
                )
                checkpoints = []
                for step in range(1, max(budgets) + 1):
                    item = choose_item_ablation(
                        policy, mean, counts, args.K, policy_rng
                    )
                    y = w_true[item] + policy_rng.normal(
                        0.0, math.sqrt(obs_var[item])
                    )
                    counts[item] += 1
                    mean, cov = update_gaussian(mean, cov, item, y, obs_var[item])
                    if step in budget_set:
                        chosen = np.argpartition(mean, -args.K)[-args.K:]
                        checkpoints.append(
                            oracle_sum - float(w_true[chosen].sum())
                        )
                records[prior_name][policy].append(checkpoints)

    rows = []
    for prior_name in priors:
        for b_idx, budget in enumerate(budgets):
            for policy in ABLATION_POLICIES:
                gaps = np.array(
                    [w[b_idx] for w in records[prior_name][policy]]
                )
                rows.append(
                    {
                        "prior": prior_name,
                        "budget": budget,
                        "policy": policy,
                        "mean_gap": round(float(gaps.mean()), 4),
                        "se": round(
                            float(gaps.std(ddof=1) / math.sqrt(len(gaps))), 4
                        ),
                        "worlds": len(gaps),
                    }
                )

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        f.write(
            f"# L={args.L} K={args.K} warm_start={args.warm_start} "
            f"budgets={budgets} seed={args.seed}; same world-generation "
            "protocol as synthetic_allocation_benchmarks.py.\n"
        )
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
