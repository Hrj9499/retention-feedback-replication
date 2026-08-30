"""Greedy versus best fixed allocation, small-scale numerical check.

Quantifies the gap between (i) the best fixed allocation of the whole
response budget found by direct numerical optimization, (ii) an exact
one-step-lookahead greedy allocation, (iii) the operating combined
cutoff x uncertainty rule from Section 6, and (iv) even allocation.

The objective is the Gaussian-proxy value E[V_K(w_tilde(N))] with
w_tilde ~ N(mu, Sigma - A(N)), estimated with common random draws so all
policies and the optimizer face the same deterministic objective.
Catalog size is deliberately small so the direct optimization is
reliable.

Outputs greedy_vs_optimal_summary.csv.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

THRESHOLDS5 = [-np.inf, -1.0, -0.2, 0.2, 1.0, np.inf]


def phi(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def Phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def fisher_info5(w: float) -> float:
    total = 0.0
    for lo, hi in zip(THRESHOLDS5[:-1], THRESHOLDS5[1:]):
        Phi_hi = 1.0 if np.isposinf(hi) else Phi(hi - w)
        Phi_lo = 0.0 if np.isneginf(lo) else Phi(lo - w)
        prob = max(Phi_hi - Phi_lo, 1e-12)
        d_hi = 0.0 if np.isposinf(hi) else phi(hi - w)
        d_lo = 0.0 if np.isneginf(lo) else phi(lo - w)
        deriv = d_lo - d_hi
        total += (deriv * deriv) / prob
    return total


def kappa5(mu: float, gh_x: np.ndarray, gh_w: np.ndarray) -> float:
    terms = [
        weight * fisher_info5(mu + math.sqrt(2.0) * x)
        for x, weight in zip(gh_x, gh_w)
    ]
    return sum(terms) / math.sqrt(math.pi)


def ar1_cov(size: int, rho: float) -> np.ndarray:
    idx = np.arange(size)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def proxy_value(
    mu: np.ndarray,
    sigma_inv: np.ndarray,
    kappas: np.ndarray,
    counts: np.ndarray,
    k: int,
    z_draws: np.ndarray,
) -> float:
    precision = sigma_inv + np.diag(kappas * np.maximum(counts, 0.0))
    a_matrix = np.linalg.inv(precision)
    sigma = np.linalg.inv(sigma_inv)
    cov = sigma - a_matrix
    # Symmetrize and clip tiny negative eigenvalues from finite arithmetic.
    cov = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(cov)
    root = vecs @ np.diag(np.sqrt(np.clip(vals, 0.0, None)))
    samples = mu[None, :] + z_draws @ root.T
    part = np.partition(samples, -k, axis=1)[:, -k:]
    return float(part.sum(axis=1).mean())


def greedy_allocation(
    mu, sigma_inv, kappas, budget, k, z_draws
) -> np.ndarray:
    counts = np.zeros(len(mu))
    base = proxy_value(mu, sigma_inv, kappas, counts, k, z_draws)
    for _ in range(budget):
        best_gain, best_j = -np.inf, 0
        for j in range(len(mu)):
            counts[j] += 1.0
            gain = proxy_value(mu, sigma_inv, kappas, counts, k, z_draws) - base
            counts[j] -= 1.0
            if gain > best_gain:
                best_gain, best_j = gain, j
        counts[best_j] += 1.0
        base += best_gain
    return counts


def combined_rule_allocation(
    mu, sigma_inv, kappas, budget, k, z_draws, bandwidth=0.5
) -> np.ndarray:
    counts = np.zeros(len(mu))
    tau = np.diag(sigma_inv)
    for _ in range(budget):
        sorted_mu = np.sort(mu)[::-1]
        cutoff = sorted_mu[k - 1]
        proximity = np.exp(-0.5 * ((mu - cutoff) / bandwidth) ** 2)
        learnability = kappas / (tau + kappas * counts) ** 2
        counts[np.argmax(proximity * learnability)] += 1.0
    return counts


def optimal_allocation(
    mu, sigma_inv, kappas, budget, k, z_draws, seed
) -> np.ndarray:
    n = len(mu)
    rng = np.random.default_rng(seed)

    def neg_value(x):
        return -proxy_value(mu, sigma_inv, kappas, x, k, z_draws)

    constraints = [{"type": "eq", "fun": lambda x: x.sum() - budget}]
    bounds = [(0.0, float(budget))] * n
    starts = [np.full(n, budget / n)]
    for _ in range(5):
        raw = rng.dirichlet(np.ones(n)) * budget
        starts.append(raw)
    best_x, best_v = None, np.inf
    for x0 in starts:
        res = minimize(
            neg_value,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 400, "ftol": 1e-9},
        )
        if res.fun < best_v:
            best_v, best_x = res.fun, res.x
    return best_x


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, default=20)
    parser.add_argument("--catalog", type=int, default=12)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--mc-draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--out", type=Path, default=Path("greedy_vs_optimal_summary.csv"))
    args = parser.parse_args()

    gh_x, gh_w = np.polynomial.hermite.hermgauss(21)
    rng = np.random.default_rng(args.seed)
    L, K = args.catalog, args.k
    budgets = [2 * L, 6 * L]
    priors = {
        "independent": np.eye(L),
        "AR(1) rho=0.6": ar1_cov(L, 0.6),
    }

    records = []
    for prior_name, sigma in priors.items():
        sigma_inv = np.linalg.inv(sigma)
        for world in range(args.worlds):
            mu = rng.standard_normal(L)
            z_draws = rng.standard_normal((args.mc_draws, L))
            kappas = np.array([kappa5(m, gh_x, gh_w) for m in mu])
            oracle = float(
                np.partition(
                    mu[None, :] + (z_draws @ np.linalg.cholesky(sigma).T),
                    -K,
                    axis=1,
                )[:, -K:].sum(axis=1).mean()
            )
            for budget in budgets:
                allocs = {
                    "even": np.full(L, budget / L),
                    "greedy_exact": greedy_allocation(
                        mu, sigma_inv, kappas, budget, K, z_draws
                    ),
                    "combined_rule": combined_rule_allocation(
                        mu, sigma_inv, kappas, budget, K, z_draws
                    ),
                    "slsqp": optimal_allocation(
                        mu, sigma_inv, kappas, budget, K, z_draws,
                        seed=args.seed + world,
                    ),
                }
                values = {
                    name: proxy_value(mu, sigma_inv, kappas, x, K, z_draws)
                    for name, x in allocs.items()
                }
                # The benchmark is the best fixed allocation found by ANY
                # method, so shortfalls are nonnegative by construction and
                # the comparison is honest when the direct optimizer stalls.
                values["best_found"] = max(values.values())
                for name, value in values.items():
                    records.append(
                        {
                            "prior": prior_name,
                            "world": world,
                            "budget": budget,
                            "policy": name,
                            "value": value,
                            "gap_vs_oracle": oracle - value,
                        }
                    )

    # Aggregate: shortfall to the best fixed allocation found, and share of
    # the achievable improvement over even allocation, per prior/budget.
    rows = []
    for prior_name in priors:
        for budget in budgets:
            sub = [
                r for r in records
                if r["prior"] == prior_name and r["budget"] == budget
            ]
            by_policy = {}
            for name in ("even", "greedy_exact", "combined_rule", "best_found"):
                vals = [r["value"] for r in sub if r["policy"] == name]
                by_policy[name] = np.array(vals)
            best = by_policy["best_found"]
            span = best - by_policy["even"]
            for name in ("greedy_exact", "combined_rule"):
                shortfall = best - by_policy[name]
                captured = 100.0 * (by_policy[name] - by_policy["even"]) / span
                rows.append(
                    {
                        "prior": prior_name,
                        "budget": budget,
                        "policy": name,
                        "mean_shortfall_vs_best": round(float(shortfall.mean()), 4),
                        "se_shortfall": round(float(shortfall.std(ddof=1) / math.sqrt(len(shortfall))), 4),
                        "mean_pct_of_achievable_improvement": round(float(captured.mean()), 1),
                        "mean_even_to_best_span": round(float(span.mean()), 4),
                        "worlds": len(shortfall),
                    }
                )

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        f.write(
            f"# L={L} K={K} budgets={budgets} mc_draws={args.mc_draws} "
            f"seed={args.seed}; optimal_fixed = SLSQP multi-start on the "
            "common-random-numbers proxy objective.\n"
        )
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
