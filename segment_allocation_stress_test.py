#!/usr/bin/env python3
"""Two-segment stress test for the finite-segment allocation formula.

Segment A has higher decision weight but a settled and well-sampled cutoff.
Segment B has lower decision weight but a crowded, weakly sampled cutoff. The
experiment checks whether segment-aware boundary scores mechanically chase the
large segment or move responses to the smaller unresolved segment.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from synthetic_allocation_benchmarks import (
    boundary_weight,
    expected_kappa,
    local_marginal_score,
    scalar_se,
    update_gaussian,
)


POLICIES = (
    "weight_only",
    "cutoff_only_unweighted",
    "segment_weighted_cutoff",
    "segment_weighted_local_marginal",
)


@dataclass
class Segment:
    name: str
    weight: float
    mean: np.ndarray
    cov: np.ndarray
    truth: np.ndarray
    kappas: np.ndarray
    counts: np.ndarray
    added: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=40)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--worlds", type=int, default=100)
    parser.add_argument("--budget", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument(
        "--output",
        default="segment_allocation_summary.csv",
        help="CSV path for the summary table.",
    )
    parser.add_argument(
        "--raw-output",
        default="segment_allocation_raw.csv",
        help="CSV path for one row per world and policy.",
    )
    return parser.parse_args()


def settled_mean(rng: np.random.Generator, size: int, k: int) -> np.ndarray:
    ordered = np.empty(size)
    ordered[:k] = np.linspace(1.7, 0.95, k)
    ordered[k:] = np.linspace(0.10, -1.8, size - k)
    ordered += rng.normal(0.0, 0.015, size)
    return ordered[rng.permutation(size)]


def crowded_mean(rng: np.random.Generator, size: int, k: int) -> np.ndarray:
    ordered = np.empty(size)
    ordered[: max(k - 3, 0)] = np.linspace(1.1, 0.55, max(k - 3, 0))
    start = max(k - 3, 0)
    end = min(k + 9, size)
    local = np.arange(start, end)
    ordered[start:end] = 0.035 * (k - local) + rng.normal(0.0, 0.012, end - start)
    ordered[end:] = np.linspace(-0.35, -1.9, size - end)
    return ordered[rng.permutation(size)]


def initialize_segment(
    name: str,
    weight: float,
    prior_mean: np.ndarray,
    prior_sd: float,
    warm_start: int,
    truth_rng: np.random.Generator,
    obs_rng: np.random.Generator,
    gh_x: np.ndarray,
    gh_w: np.ndarray,
) -> Segment:
    cov = np.eye(len(prior_mean)) * prior_sd * prior_sd
    truth = truth_rng.multivariate_normal(prior_mean, cov)
    kappas = np.array(
        [expected_kappa(float(mu), gh_x, gh_w) for mu in prior_mean],
        dtype=float,
    )
    obs_var = 1.0 / np.clip(kappas, 1e-8, None)
    mean = prior_mean.copy()
    post_cov = cov.copy()
    counts = np.zeros(len(prior_mean), dtype=int)
    for item in range(len(prior_mean)):
        for _ in range(warm_start):
            y = truth[item] + obs_rng.normal(0.0, math.sqrt(obs_var[item]))
            counts[item] += 1
            mean, post_cov = update_gaussian(mean, post_cov, item, y, obs_var[item])
    return Segment(name, weight, mean, post_cov, truth, kappas, counts)


def weighted_gap(segments: list[Segment], k: int) -> float:
    total = 0.0
    for segment in segments:
        oracle = float(np.sort(segment.truth)[-k:].sum())
        chosen = np.argpartition(segment.mean, -k)[-k:]
        total += segment.weight * (oracle - float(segment.truth[chosen].sum()))
    return total


def choose_weight_only(segments: list[Segment], rng: np.random.Generator) -> tuple[int, int]:
    total_added = sum(segment.added for segment in segments)
    if total_added == 0:
        seg_idx = int(np.argmax([segment.weight for segment in segments]))
    else:
        deficits = [
            segment.weight - (segment.added / max(total_added, 1))
            for segment in segments
        ]
        seg_idx = int(np.argmax(deficits))
    segment = segments[seg_idx]
    eligible = np.flatnonzero(segment.counts == segment.counts.min())
    return seg_idx, int(rng.choice(eligible))


def choose_scored(
    policy: str,
    segments: list[Segment],
    k: int,
    rng: np.random.Generator,
) -> tuple[int, int]:
    best_score = -np.inf
    best_pairs: list[tuple[int, int]] = []

    for seg_idx, segment in enumerate(segments):
        if policy == "cutoff_only_unweighted":
            scores = boundary_weight(segment.mean, k, floor=0.0) / (
                1.0 + 0.10 * segment.counts
            )
        elif policy == "segment_weighted_cutoff":
            scores = segment.weight * boundary_weight(segment.mean, k, floor=0.0) / (
                1.0 + 0.10 * segment.counts
            )
        elif policy == "segment_weighted_local_marginal":
            scores = segment.weight * local_marginal_score(
                segment.mean,
                segment.cov,
                segment.kappas,
                k,
            )
        else:
            raise ValueError(f"Unknown policy: {policy}")

        local_best = float(scores.max())
        if local_best > best_score + 1e-12:
            best_score = local_best
            best_pairs = [(seg_idx, int(item)) for item in np.flatnonzero(np.isclose(scores, local_best))]
        elif np.isclose(local_best, best_score):
            best_pairs.extend(
                (seg_idx, int(item)) for item in np.flatnonzero(np.isclose(scores, local_best))
            )

    return best_pairs[int(rng.integers(0, len(best_pairs)))]


def run_policy(
    base_segments: list[Segment],
    policy: str,
    budget: int,
    k: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    segments = [
        Segment(
            segment.name,
            segment.weight,
            segment.mean.copy(),
            segment.cov.copy(),
            segment.truth.copy(),
            segment.kappas.copy(),
            segment.counts.copy(),
            0,
        )
        for segment in base_segments
    ]

    for _ in range(budget):
        if policy == "weight_only":
            seg_idx, item = choose_weight_only(segments, rng)
        else:
            seg_idx, item = choose_scored(policy, segments, k, rng)
        segment = segments[seg_idx]
        obs_var = 1.0 / max(float(segment.kappas[item]), 1e-8)
        y = segment.truth[item] + rng.normal(0.0, math.sqrt(obs_var))
        segment.counts[item] += 1
        segment.added += 1
        segment.mean, segment.cov = update_gaussian(
            segment.mean,
            segment.cov,
            item,
            y,
            obs_var,
        )

    smaller_segment_share = next(
        segment.added / max(budget, 1)
        for segment in segments
        if segment.name == "B"
    )
    return weighted_gap(segments, k), smaller_segment_share


def summarize(values: np.ndarray) -> tuple[float, float]:
    return float(values.mean()), scalar_se(values)


def main() -> None:
    args = parse_args()
    gh_x, gh_w = np.polynomial.hermite.hermgauss(31)
    master_rng = np.random.default_rng(args.seed)
    raw_rows = []
    records = {policy: {"gap": [], "share_b": []} for policy in POLICIES}

    for world in range(args.worlds):
        world_seed = int(master_rng.integers(0, 2**32 - 1))
        world_rng = np.random.default_rng(world_seed)
        mean_a = settled_mean(world_rng, args.L, args.K)
        mean_b = crowded_mean(world_rng, args.L, args.K)
        base_segments = [
            initialize_segment(
                "A",
                0.70,
                mean_a,
                prior_sd=0.35,
                warm_start=20,
                truth_rng=world_rng,
                obs_rng=world_rng,
                gh_x=gh_x,
                gh_w=gh_w,
            ),
            initialize_segment(
                "B",
                0.30,
                mean_b,
                prior_sd=0.95,
                warm_start=2,
                truth_rng=world_rng,
                obs_rng=world_rng,
                gh_x=gh_x,
                gh_w=gh_w,
            ),
        ]

        for policy in POLICIES:
            policy_rng = np.random.default_rng(int(world_rng.integers(0, 2**32 - 1)))
            gap, share_b = run_policy(base_segments, policy, args.budget, args.K, policy_rng)
            records[policy]["gap"].append(gap)
            records[policy]["share_b"].append(share_b)
            raw_rows.append(
                {
                    "world": world,
                    "world_seed": world_seed,
                    "policy": policy,
                    "budget": args.budget,
                    "gap": gap,
                    "share_to_smaller_crowded_segment": share_b,
                }
            )

    gap_arrays = {policy: np.array(records[policy]["gap"], dtype=float) for policy in POLICIES}
    share_arrays = {policy: np.array(records[policy]["share_b"], dtype=float) for policy in POLICIES}
    min_gap = np.vstack([gap_arrays[policy] for policy in POLICIES]).min(axis=0)

    rows = []
    for policy in POLICIES:
        gap_mean, gap_se = summarize(gap_arrays[policy])
        share_mean, share_se = summarize(share_arrays[policy])
        improvement = gap_arrays["weight_only"] - gap_arrays[policy]
        rows.append(
            {
                "policy": policy,
                "worlds": args.worlds,
                "budget": args.budget,
                "mean_weighted_gap": gap_mean,
                "weighted_gap_se": gap_se,
                "mean_share_to_smaller_crowded_segment": share_mean,
                "share_to_smaller_crowded_segment_se": share_se,
                "improvement_vs_weight_only_mean": float(improvement.mean()),
                "improvement_vs_weight_only_se": scalar_se(improvement),
                "win_rate": float(np.isclose(gap_arrays[policy], min_gap).mean()),
            }
        )

    with Path(args.raw_output).open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "world",
                "world_seed",
                "policy",
                "budget",
                "gap",
                "share_to_smaller_crowded_segment",
            ],
        )
        writer.writeheader()
        writer.writerows(raw_rows)

    with Path(args.output).open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "policy",
                "worlds",
                "budget",
                "mean_weighted_gap",
                "weighted_gap_se",
                "mean_share_to_smaller_crowded_segment",
                "share_to_smaller_crowded_segment_se",
                "improvement_vs_weight_only_mean",
                "improvement_vs_weight_only_se",
                "win_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
