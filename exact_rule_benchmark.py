"""Exact theory rule and SAA reference at experimental scale.

Answers the referee question: does the theory-derived rule lose to the
guarded cutoff heuristic because of the theory, or because of the kernel
approximation? Under the independent prior, the exact one-step value of
the next answer has a closed form (no kernel, no Monte Carlo per item):
with current mean m_j, variance v_j, and per-answer precision kappa_j,
one answer moves m_j to a normal with standard deviation
s_j = v_j / sqrt(v_j + 1/kappa_j), and the expected top-K value after
the answer is E[max(t_j, m_j')] plus the fixed top-(K-1) rest, where
t_j is the K-th largest of the other means. The exact rule assigns each
answer to the item with the largest closed-form one-step gain. This is
Proposition 4 evaluated exactly.

The script also computes an SAA reference: the best fixed (static)
allocation of the whole budget from the post-warm-start belief,
optimized by multi-start SLSQP on a common-random-numbers preposterior
objective, then evaluated on one simulated realization per world (same
interface as the adaptive policies). Note the reference is static:
adaptive rules may exceed it.

World generation replicates synthetic_allocation_benchmarks.py exactly
(same master seed, same warm start), restricted to the independent
prior where the closed form applies. Outputs exact_rule_summary.csv.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from synthetic_allocation_benchmarks import (
    ar1_cov,
    boundary_weight,
    expected_kappa,
)


def phi(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def exact_one_step_gain(m, v, kappas, k):
    """Closed-form expected one-step top-K gain for each item (vectorized)."""
    from scipy.stats import norm

    L = len(m)
    order = np.argsort(m)[::-1]
    m_k = m[order[k - 1]]
    m_k1 = m[order[k]] if k < L else -np.inf
    t = np.full(L, m_k)
    t[order[:k]] = m_k1
    s = v / np.sqrt(v + 1.0 / kappas)
    d = (m - t) / np.maximum(s, 1e-300)
    e_max = t * norm.cdf(-d) + m * norm.cdf(d) + s * norm.pdf(d)
    return e_max - np.maximum(t, m)


def run_exact_policy(m0, prec0, kappas, w_true, budgets, k, rng):
    m = m0.copy()
    prec = prec0.copy()
    checkpoints = []
    budget_set = set(budgets)
    for step in range(1, max(budgets) + 1):
        v = 1.0 / prec
        gains = exact_one_step_gain(m, v, kappas, k)
        j = int(np.argmax(gains))
        y = w_true[j] + rng.normal(0.0, math.sqrt(1.0 / kappas[j]))
        m[j] = (prec[j] * m[j] + kappas[j] * y) / (prec[j] + kappas[j])
        prec[j] += kappas[j]
        if step in budget_set:
            chosen = np.argpartition(m, -k)[-k:]
            checkpoints.append(float(np.sort(w_true)[-k:].sum() - w_true[chosen].sum()))
    return checkpoints


def run_guarded_cutoff(m0, prec0, kappas, w_true, budgets, k, rng):
    m = m0.copy()
    prec = prec0.copy()
    counts = np.zeros(len(m0))
    checkpoints = []
    budget_set = set(budgets)
    for step in range(1, max(budgets) + 1):
        scores = boundary_weight(m, k) / (1.0 + 0.10 * counts)
        best = np.flatnonzero(np.isclose(scores, scores.max()))
        j = int(rng.choice(best))
        y = w_true[j] + rng.normal(0.0, math.sqrt(1.0 / kappas[j]))
        m[j] = (prec[j] * m[j] + kappas[j] * y) / (prec[j] + kappas[j])
        prec[j] += kappas[j]
        counts[j] += 1.0
        if step in budget_set:
            chosen = np.argpartition(m, -k)[-k:]
            checkpoints.append(float(np.sort(w_true)[-k:].sum() - w_true[chosen].sum()))
    return checkpoints


def saa_static_gap(m0, prec0, kappas, w_true, budget, k, rng, draws, starts):
    L = len(m0)
    v0 = 1.0 / prec0
    z = rng.standard_normal((draws, L))

    def neg_value(x):
        x = np.maximum(x, 0.0)
        post_var = 1.0 / (prec0 + kappas * x)
        pre_sd = np.sqrt(np.maximum(v0 - post_var, 0.0))
        samples = m0[None, :] + z * pre_sd[None, :]
        part = np.partition(samples, -k, axis=1)[:, -k:]
        return -float(part.sum(axis=1).mean())

    cons = [{"type": "eq", "fun": lambda x: x.sum() - budget}]
    bounds = [(0.0, float(budget))] * L
    best_x, best_v = np.full(L, budget / L), neg_value(np.full(L, budget / L))
    x0s = [np.full(L, budget / L)]
    for _ in range(starts - 1):
        x0s.append(rng.dirichlet(np.ones(L)) * budget)
    for x0 in x0s:
        res = minimize(
            neg_value, x0, method="SLSQP", bounds=bounds, constraints=cons,
            options={"maxiter": 150, "ftol": 1e-8},
        )
        if res.fun < best_v:
            best_v, best_x = res.fun, np.maximum(res.x, 0.0)
    # Evaluate one simulated realization of the chosen static allocation.
    n_j = best_x
    y_mean_var = np.where(n_j > 1e-9, 1.0 / (kappas * np.maximum(n_j, 1e-9)), 0.0)
    y_bar = w_true + rng.standard_normal(L) * np.sqrt(y_mean_var)
    m_post = np.where(
        n_j > 1e-9,
        (prec0 * m0 + kappas * n_j * y_bar) / (prec0 + kappas * n_j),
        m0,
    )
    chosen = np.argpartition(m_post, -k)[-k:]
    return float(np.sort(w_true)[-k:].sum() - w_true[chosen].sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, default=100)
    parser.add_argument("--saa-worlds", type=int, default=30)
    parser.add_argument("--L", type=int, default=60)
    parser.add_argument("--K", type=int, default=6)
    parser.add_argument("--warm-start", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260406)
    parser.add_argument("--saa-draws", type=int, default=2000)
    parser.add_argument("--saa-starts", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("exact_rule_summary.csv"))
    args = parser.parse_args()

    L, K = args.L, args.K
    budgets = [10 * L, 30 * L, 100 * L]
    gh_x, gh_w = np.polynomial.hermite.hermgauss(31)
    master_rng = np.random.default_rng(args.seed)

    # Replicate the published world sequence: the main benchmark iterates
    # AR(1) first, then independent; consume the AR(1) world seeds so the
    # independent worlds match the published ones.
    _ = [int(master_rng.integers(0, 2**32 - 1)) for _ in range(args.worlds)]

    results = {"guarded_cutoff": [], "exact_one_step": []}
    saa_gaps = {b: [] for b in budgets}
    for world in range(args.worlds):
        world_seed = int(master_rng.integers(0, 2**32 - 1))
        world_rng = np.random.default_rng(world_seed)
        prior_mean = world_rng.normal(0.0, 1.0, size=L)
        w_true = world_rng.multivariate_normal(prior_mean, np.eye(L))
        kappas = np.array([expected_kappa(float(mu), gh_x, gh_w) for mu in prior_mean])
        obs_var = 1.0 / np.clip(kappas, 1e-8, None)

        m0 = prior_mean.copy()
        prec0 = np.ones(L)
        for item in range(L):
            for _ in range(args.warm_start):
                y = w_true[item] + world_rng.normal(0.0, math.sqrt(obs_var[item]))
                m0[item] = (prec0[item] * m0[item] + kappas[item] * y) / (
                    prec0[item] + kappas[item]
                )
                prec0[item] += kappas[item]

        for name, runner in (
            ("guarded_cutoff", run_guarded_cutoff),
            ("exact_one_step", run_exact_policy),
        ):
            policy_rng = np.random.default_rng(int(world_rng.integers(0, 2**32 - 1)))
            results[name].append(
                runner(m0, prec0, kappas, w_true, budgets, K, policy_rng)
            )
        if world < args.saa_worlds:
            saa_rng = np.random.default_rng(int(world_rng.integers(0, 2**32 - 1)))
            for budget in budgets:
                saa_gaps[budget].append(
                    saa_static_gap(
                        m0, prec0, kappas, w_true, budget, K, saa_rng,
                        args.saa_draws, args.saa_starts,
                    )
                )
        if (world + 1) % 10 == 0:
            print(f"world {world + 1}/{args.worlds} done", flush=True)

    rows = []
    for name, worlds in results.items():
        arr = np.array(worlds)
        for b_idx, budget in enumerate(budgets):
            rows.append(
                {
                    "policy": name,
                    "budget": budget,
                    "mean_gap": round(float(arr[:, b_idx].mean()), 4),
                    "se": round(
                        float(arr[:, b_idx].std(ddof=1) / math.sqrt(arr.shape[0])), 4
                    ),
                    "worlds": arr.shape[0],
                }
            )
    # 30-world-subset rows: policies restricted to the SAA worlds, plus the
    # paired guarded-minus-SAA difference on those same worlds.
    n_sub = args.saa_worlds
    for name, worlds in results.items():
        arr = np.array(worlds)[:n_sub]
        for b_idx, budget in enumerate(budgets):
            rows.append(
                {
                    "policy": name + "_subset",
                    "budget": budget,
                    "mean_gap": round(float(arr[:, b_idx].mean()), 4),
                    "se": round(
                        float(arr[:, b_idx].std(ddof=1) / math.sqrt(arr.shape[0])), 4
                    ),
                    "worlds": arr.shape[0],
                }
            )
    guarded_sub = np.array(results["guarded_cutoff"])[:n_sub]
    for b_idx, budget in enumerate(budgets):
        diff = guarded_sub[:, b_idx] - np.array(saa_gaps[budget])
        rows.append(
            {
                "policy": "paired_guarded_minus_saa",
                "budget": budget,
                "mean_gap": round(float(diff.mean()), 4),
                "se": round(float(diff.std(ddof=1) / math.sqrt(len(diff))), 4),
                "worlds": len(diff),
            }
        )
    for budget in budgets:
        arr = np.array(saa_gaps[budget])
        rows.append(
            {
                "policy": "saa_static",
                "budget": budget,
                "mean_gap": round(float(arr.mean()), 4),
                "se": round(float(arr.std(ddof=1) / math.sqrt(len(arr))), 4),
                "worlds": len(arr),
            }
        )

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        f.write(
            f"# independent prior only; L={L} K={K} warm_start={args.warm_start} "
            f"budgets={budgets} seed={args.seed}; worlds match the published "
            "independent-prior worlds of synthetic_allocation_benchmarks.py; "
            "saa_static is a static (non-adaptive) reference.\n"
        )
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
