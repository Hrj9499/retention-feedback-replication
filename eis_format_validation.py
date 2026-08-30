#!/usr/bin/env python3
"""Synthetic validation of EIS as a format-screening rule.

This script compares:
1. The format ranking induced by EIS(Q,S)=beta(S)*Q*alpha(Q,S), and
2. The realized ranking under the nonfactorized kappa_i(S) values,
   using the cutoff-aware allocation heuristic and the Gaussian proxy
   objective from the paper.

The default candidate formats and burden scenarios mirror Table 1.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter

import numpy as np


FORMATS = [
    ("Binary, Q=3", 3, 2),
    ("Binary, Q=5", 5, 2),
    ("5-point, Q=3", 3, 5),
    ("5-point, Q=5", 5, 5),
    ("5-point, Q=7", 7, 5),
]

SCENARIOS = {
    "Low burden sensitivity": {
        "Binary, Q=3": 0.90,
        "Binary, Q=5": 0.84,
        "5-point, Q=3": 0.86,
        "5-point, Q=5": 0.78,
        "5-point, Q=7": 0.69,
    },
    "Moderate burden sensitivity": {
        "Binary, Q=3": 0.82,
        "Binary, Q=5": 0.68,
        "5-point, Q=3": 0.74,
        "5-point, Q=5": 0.52,
        "5-point, Q=7": 0.36,
    },
    "High burden sensitivity": {
        "Binary, Q=3": 0.80,
        "Binary, Q=5": 0.50,
        "5-point, Q=3": 0.58,
        "5-point, Q=5": 0.33,
        "5-point, Q=7": 0.17,
    },
}

THRESHOLDS = {
    2: [-np.inf, 0.0, np.inf],
    5: [-np.inf, -1.0, -0.2, 0.2, 1.0, np.inf],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=60)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--rho", type=float, default=0.6)
    parser.add_argument("--T", type=int, default=30)
    parser.add_argument("--worlds", type=int, default=30)
    parser.add_argument("--mc-draws", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output",
        default="eis_format_validation_summary.csv",
        help="CSV path for the scenario-level summary.",
    )
    parser.add_argument(
        "--raw-output",
        default="eis_format_validation_raw.csv",
        help="CSV path for one row per scenario, world, and candidate format.",
    )
    return parser.parse_args()


def phi(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def Phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def fisher_info(w: float, scale: int) -> float:
    thresholds = THRESHOLDS[scale]
    total = 0.0
    for lo, hi in zip(thresholds[:-1], thresholds[1:]):
        Phi_hi = 1.0 if np.isposinf(hi) else Phi(hi - w)
        Phi_lo = 0.0 if np.isneginf(lo) else Phi(lo - w)
        prob = max(Phi_hi - Phi_lo, 1e-12)
        d_hi = 0.0 if np.isposinf(hi) else phi(hi - w)
        d_lo = 0.0 if np.isneginf(lo) else phi(lo - w)
        deriv = d_lo - d_hi
        total += (deriv * deriv) / prob
    return total


def kappa(mu: float, scale: int, gh_x: np.ndarray, gh_w: np.ndarray) -> float:
    terms = [
        weight * fisher_info(mu + math.sqrt(2.0) * x, scale)
        for x, weight in zip(gh_x, gh_w)
    ]
    return sum(terms) / math.sqrt(math.pi)


def ar1_cov(size: int, rho: float) -> np.ndarray:
    idx = np.arange(size)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def allocate_budget(
    mu: np.ndarray,
    sigma: np.ndarray,
    kappas: np.ndarray,
    budget: int,
    top_k: int,
) -> np.ndarray:
    tau = np.diag(np.linalg.inv(sigma))
    mu_sorted = np.sort(mu)[::-1]
    cutoff = mu_sorted[top_k - 1]
    next_idx = min(top_k, len(mu_sorted) - 1)
    bandwidth = max(0.15, 2.0 * abs(mu_sorted[top_k - 1] - mu_sorted[next_idx]) + 0.05)
    cutoff_score = np.exp(-((np.abs(mu - cutoff) / bandwidth) ** 2))

    counts = np.zeros(len(mu), dtype=int)
    for _ in range(int(max(0, budget))):
        priorities = cutoff_score * kappas / (tau + kappas * counts) ** 2
        counts[int(np.argmax(priorities))] += 1
    return counts


def topk_proxy_value(
    mu: np.ndarray,
    sigma: np.ndarray,
    kappas: np.ndarray,
    counts: np.ndarray,
    top_k: int,
    z_draws: np.ndarray,
) -> float:
    precision = np.linalg.inv(sigma) + np.diag(kappas * counts)
    posterior_cov = np.linalg.inv(precision)
    learned_cov = sigma - posterior_cov
    learned_cov = (learned_cov + learned_cov.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(learned_cov)
    eigvals = np.clip(eigvals, 0.0, None)
    samples = mu[:, None] + (eigvecs @ np.diag(np.sqrt(eigvals))) @ z_draws
    partitioned = np.partition(samples, samples.shape[0] - top_k, axis=0)
    return float(partitioned[-top_k:, :].sum(axis=0).mean())


def spearman_rho(values1: list[float], values2: list[float]) -> float:
    order1 = np.argsort(-np.array(values1))
    order2 = np.argsort(-np.array(values2))
    rank1 = np.empty_like(order1)
    rank2 = np.empty_like(order2)
    rank1[order1] = np.arange(len(values1))
    rank2[order2] = np.arange(len(values2))
    diff = rank1 - rank2
    n = len(values1)
    return float(1.0 - 6.0 * np.sum(diff * diff) / (n * (n * n - 1)))


def descending_ranks(values: list[float]) -> np.ndarray:
    order = np.argsort(-np.array(values))
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    gh_x, gh_w = np.polynomial.hermite.hermgauss(31)
    sigma = ar1_cov(args.L, args.rho)

    rows = []
    raw_rows = []
    for scenario_name, alpha_map in SCENARIOS.items():
        hit_rates = []
        near_best_rates = []
        regrets = []
        rank_corrs = []
        selected_labels = []

        for world in range(args.worlds):
            mu = rng.normal(0.0, 1.0, size=args.L)
            kappas_by_scale = {
                scale: np.array([kappa(float(m), scale, gh_x, gh_w) for m in mu])
                for scale in (2, 5)
            }
            betas = {scale: float(kappas.mean()) for scale, kappas in kappas_by_scale.items()}
            z_draws = rng.normal(size=(args.L, args.mc_draws))

            eis_scores = []
            realized_values = []
            labels = []
            for label, questions, scale in FORMATS:
                labels.append(label)
                eis_scores.append(betas[scale] * questions * alpha_map[label])
                budget = int(round(args.T * questions * alpha_map[label]))
                counts = allocate_budget(mu, sigma, kappas_by_scale[scale], budget, args.K)
                realized_values.append(
                    topk_proxy_value(mu, sigma, kappas_by_scale[scale], counts, args.K, z_draws)
                )

            eis_best = int(np.argmax(eis_scores))
            realized_best = int(np.argmax(realized_values))
            best_value = realized_values[realized_best]
            selected_value = realized_values[eis_best]
            eis_ranks = descending_ranks(eis_scores)
            realized_ranks = descending_ranks(realized_values)

            selected_labels.append(labels[eis_best])
            hit_rates.append(float(eis_best == realized_best))
            near_best_rates.append(float(selected_value >= 0.95 * best_value))
            regrets.append(float(100.0 * (best_value - selected_value) / best_value))
            rank_corrs.append(spearman_rho(eis_scores, realized_values))

            for idx, (label, questions, scale) in enumerate(FORMATS):
                format_regret_pct = 100.0 * (best_value - realized_values[idx]) / best_value
                raw_rows.append(
                    {
                        "scenario": scenario_name,
                        "world": world,
                        "format": label,
                        "questions": questions,
                        "scale": scale,
                        "completion_rate": alpha_map[label],
                        "beta_scale": betas[scale],
                        "eis_score": eis_scores[idx],
                        "realized_proxy_value": realized_values[idx],
                        "budget": int(round(args.T * questions * alpha_map[label])),
                        "eis_rank": int(eis_ranks[idx]),
                        "realized_rank": int(realized_ranks[idx]),
                        "selected_by_eis": int(idx == eis_best),
                        "ex_post_best": int(idx == realized_best),
                        "best_realized_proxy_value": best_value,
                        "format_regret_pct": format_regret_pct,
                        "selected_format": labels[eis_best],
                        "ex_post_best_format": labels[realized_best],
                        "eis_selected_hit": int(eis_best == realized_best),
                        "eis_selected_within_5pct": int(selected_value >= 0.95 * best_value),
                        "eis_selected_regret_pct": 100.0 * (best_value - selected_value) / best_value,
                        "scenario_spearman_rho": spearman_rho(eis_scores, realized_values),
                    }
                )

        rows.append(
            {
                "scenario": scenario_name,
                "eis_selected_format": Counter(selected_labels).most_common(1)[0][0],
                "hit_rate": np.mean(hit_rates),
                "within_5pct_of_best": np.mean(near_best_rates),
                "mean_regret_pct": np.mean(regrets),
                "mean_spearman_rho": np.mean(rank_corrs),
            }
        )

    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario",
                "eis_selected_format",
                "hit_rate",
                "within_5pct_of_best",
                "mean_regret_pct",
                "mean_spearman_rho",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(args.raw_output, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario",
                "world",
                "format",
                "questions",
                "scale",
                "completion_rate",
                "beta_scale",
                "eis_score",
                "realized_proxy_value",
                "budget",
                "eis_rank",
                "realized_rank",
                "selected_by_eis",
                "ex_post_best",
                "best_realized_proxy_value",
                "format_regret_pct",
                "selected_format",
                "ex_post_best_format",
                "eis_selected_hit",
                "eis_selected_within_5pct",
                "eis_selected_regret_pct",
                "scenario_spearman_rho",
            ],
        )
        writer.writeheader()
        writer.writerows(raw_rows)

    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
