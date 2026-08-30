#!/usr/bin/env python3
"""MovieLens multi-split robustness for the cutoff-targeting mechanism.

The baseline MovieLens exercise uses one clean chronological split. This
script repeats the same history/catalog/feedback/holdout protocol across
multiple chronological cut points and feedback/evaluation split points, then
summarizes variation across split configurations.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from movielens_proxy_benchmarks import (
    build_future_orders,
    prepare_dataset,
    resolve_data_path,
    run_policy,
    summarize_runs,
)


CONFIGS = (
    ("h15_f50", 200, 10, 0.15, 0.50),
    ("h20_f50", 200, 10, 0.20, 0.50),
    ("h25_f50", 200, 10, 0.25, 0.50),
    ("h30_f50", 200, 10, 0.30, 0.50),
    ("h20_f40", 200, 10, 0.20, 0.40),
    ("h20_f60", 200, 10, 0.20, 0.60),
)
POLICIES = ("uniform", "uncertainty_only", "cutoff_only", "cutoff_x_uncertainty")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help=(
            "Path to MovieLens u.data. If omitted, the script downloads/uses "
            "data/ml-100k/u.data. Explicit paths are checked against the "
            "expected MovieLens 100K SHA256."
        ),
    )
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[1000, 2000],
        help="Budgets to summarize for each split.",
    )
    parser.add_argument(
        "--aggregate-output",
        default="movielens_multisplit_aggregate.csv",
        help="CSV path for aggregate results across split configurations.",
    )
    parser.add_argument(
        "--split-output",
        default="movielens_multisplit_summary.csv",
        help="CSV path for one row per split configuration and budget.",
    )
    parser.add_argument(
        "--raw-output",
        default="movielens_multisplit_raw.csv",
        help="CSV path for one row per split configuration, repetition, policy, and budget.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not download MovieLens data; fail if --data-path or data/ml-100k/u.data is unavailable.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify all configured clean splits, then exit without running policy simulations.",
    )
    return parser.parse_args()


def scalar_se(values: pd.Series) -> float:
    if len(values) <= 1:
        return float("nan")
    return float(values.std(ddof=1) / math.sqrt(len(values)))


def run_config(
    data_path: Path,
    label: str,
    top_l: int,
    k: int,
    history_share: float,
    feedback_share: float,
    budgets: list[int],
    reps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    (
        history_counts,
        history_sums,
        feedback_ratings,
        oracle_means,
        split_info,
        _selected_catalog,
    ) = prepare_dataset(
        data_path,
        top_l=top_l,
        history_share=history_share,
        feedback_share=feedback_share,
    )

    master_rng = np.random.default_rng(seed)
    records: dict[str, list[list[float]]] = {policy: [] for policy in POLICIES}
    raw_rows: list[dict[str, object]] = []

    for rep in range(reps):
        rep_seed = int(master_rng.integers(0, 2**32 - 1))
        future_orders = build_future_orders(feedback_ratings, np.random.default_rng(rep_seed))
        for offset, policy in enumerate(POLICIES):
            policy_rng = np.random.default_rng(rep_seed + 1000 * (offset + 1))
            gaps = run_policy(
                policy=policy,
                history_counts=history_counts,
                history_sums=history_sums,
                future_orders=future_orders,
                oracle_means=oracle_means,
                budgets=budgets,
                k=k,
                rng=policy_rng,
            )
            records[policy].append(gaps)
            for budget, gap in zip(budgets, gaps):
                raw_rows.append(
                    {
                        "config": label,
                        "rep": rep,
                        "rep_seed": rep_seed,
                        "policy": policy,
                        "ratings_used": budget,
                        "gap": gap,
                    }
                )

    summary = summarize_runs(records, budgets)
    for key, value in asdict(split_info).items():
        summary[key] = value
    summary["config"] = label
    summary["top_l"] = top_l
    summary["k"] = k
    summary["reps"] = reps
    summary["cutoff_only_reduction_vs_uniform_pct"] = 100.0 * (
        summary["uniform_mean"] - summary["cutoff_only_mean"]
    ) / summary["uniform_mean"].clip(lower=1e-8)
    summary["cutoff_x_uncertainty_reduction_vs_uniform_pct"] = 100.0 * (
        summary["uniform_mean"] - summary["cutoff_x_uncertainty_mean"]
    ) / summary["uniform_mean"].clip(lower=1e-8)
    summary["best_cutoff_aware_policy"] = np.select(
        [
            np.isclose(
                summary["cutoff_only_mean"],
                summary["cutoff_x_uncertainty_mean"],
                atol=1e-10,
            ),
            summary["cutoff_x_uncertainty_mean"] < summary["cutoff_only_mean"],
        ],
        ["Tie", "cutoff_x_uncertainty"],
        default="cutoff_only",
    )
    summary["best_cutoff_aware_beats_uniform"] = (
        summary["best_cutoff_aware_mean"] < summary["uniform_mean"]
    )
    summary["cutoff_only_beats_uniform"] = summary["cutoff_only_mean"] < summary["uniform_mean"]
    summary["cutoff_x_uncertainty_beats_uniform"] = (
        summary["cutoff_x_uncertainty_mean"] < summary["uniform_mean"]
    )

    raw = pd.DataFrame(raw_rows)
    split_cols = [
        "config",
        "ratings_used",
        "top_l",
        "k",
        "history_share",
        "feedback_share_of_post_history",
        "reps",
        "catalog_history_ratings",
        "catalog_feedback_ratings",
        "catalog_evaluation_ratings",
        "min_feedback_ratings_per_item",
        "min_evaluation_ratings_per_item",
        "uniform_mean",
        "uniform_sd",
        "uncertainty_only_mean",
        "uncertainty_only_sd",
        "cutoff_only_mean",
        "cutoff_only_sd",
        "cutoff_x_uncertainty_mean",
        "cutoff_x_uncertainty_sd",
        "cutoff_only_vs_uniform_paired_improvement_mean",
        "cutoff_only_vs_uniform_paired_improvement_se",
        "cutoff_only_vs_uniform_win_rate",
        "cutoff_x_uncertainty_vs_uniform_paired_improvement_mean",
        "cutoff_x_uncertainty_vs_uniform_paired_improvement_se",
        "cutoff_x_uncertainty_vs_uniform_win_rate",
        "cutoff_x_uncertainty_vs_cutoff_only_paired_improvement_mean",
        "cutoff_x_uncertainty_vs_cutoff_only_paired_improvement_se",
        "cutoff_x_uncertainty_vs_cutoff_only_win_rate",
        "cutoff_only_reduction_vs_uniform_pct",
        "cutoff_x_uncertainty_reduction_vs_uniform_pct",
        "best_cutoff_aware_mean",
        "best_cutoff_aware_improvement_vs_uniform_%",
        "best_cutoff_aware_policy",
        "best_cutoff_aware_beats_uniform",
        "cutoff_only_beats_uniform",
        "cutoff_x_uncertainty_beats_uniform",
    ]
    return summary[split_cols], raw


def aggregate_splits(split_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for budget, group in split_summary.groupby("ratings_used", sort=True):
        best_reduction = group["best_cutoff_aware_improvement_vs_uniform_%"]
        cutoff_reduction = group["cutoff_only_reduction_vs_uniform_pct"]
        combined_reduction = group["cutoff_x_uncertainty_reduction_vs_uniform_pct"]
        rows.append(
            {
                "ratings_used": budget,
                "split_configs": len(group),
                "reps_per_split": int(group["reps"].iloc[0]),
                "uniform_mean_across_splits": group["uniform_mean"].mean(),
                "cutoff_only_mean_across_splits": group["cutoff_only_mean"].mean(),
                "cutoff_x_uncertainty_mean_across_splits": group[
                    "cutoff_x_uncertainty_mean"
                ].mean(),
                "best_cutoff_aware_mean_across_splits": group[
                    "best_cutoff_aware_mean"
                ].mean(),
                "cutoff_only_reduction_vs_uniform_pct_mean": cutoff_reduction.mean(),
                "cutoff_only_reduction_vs_uniform_pct_se_across_splits": scalar_se(
                    cutoff_reduction
                ),
                "cutoff_x_uncertainty_reduction_vs_uniform_pct_mean": combined_reduction.mean(),
                "cutoff_x_uncertainty_reduction_vs_uniform_pct_se_across_splits": scalar_se(
                    combined_reduction
                ),
                "best_cutoff_aware_reduction_vs_uniform_pct_mean": best_reduction.mean(),
                "best_cutoff_aware_reduction_vs_uniform_pct_se_across_splits": scalar_se(
                    best_reduction
                ),
                "best_cutoff_aware_reduction_vs_uniform_pct_min": best_reduction.min(),
                "best_cutoff_aware_reduction_vs_uniform_pct_max": best_reduction.max(),
                "share_splits_best_cutoff_aware_beats_uniform": group[
                    "best_cutoff_aware_beats_uniform"
                ].mean(),
                "share_splits_cutoff_only_beats_uniform": group[
                    "cutoff_only_beats_uniform"
                ].mean(),
                "share_splits_cutoff_x_uncertainty_beats_uniform": group[
                    "cutoff_x_uncertainty_beats_uniform"
                ].mean(),
                "share_splits_cutoff_x_uncertainty_best": (
                    group["best_cutoff_aware_policy"] == "cutoff_x_uncertainty"
                ).mean(),
                "share_splits_cutoff_only_best": (
                    group["best_cutoff_aware_policy"] == "cutoff_only"
                ).mean(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    data_path = resolve_data_path(root, args.data_path, skip_download=args.skip_download)

    if args.verify_only:
        print(f"Verified MovieLens data: {data_path}")
        for label, top_l, k, history_share, feedback_share in CONFIGS:
            (
                _history_counts,
                _history_sums,
                _feedback_ratings,
                oracle_means,
                split_info,
                _selected_catalog,
            ) = prepare_dataset(
                data_path,
                top_l=top_l,
                history_share=history_share,
                feedback_share=feedback_share,
            )
            print(
                f"{label}: L={len(oracle_means)}, K={k}, "
                f"h={history_share:.2f}, f={feedback_share:.2f}, "
                f"history={split_info.catalog_history_ratings}, "
                f"feedback={split_info.catalog_feedback_ratings}, "
                f"evaluation={split_info.catalog_evaluation_ratings}, "
                f"min feedback/evaluation="
                f"{split_info.min_feedback_ratings_per_item}/"
                f"{split_info.min_evaluation_ratings_per_item}."
            )
        return

    split_frames = []
    raw_frames = []
    for idx, (label, top_l, k, history_share, feedback_share) in enumerate(CONFIGS):
        split_summary, raw = run_config(
            data_path=data_path,
            label=label,
            top_l=top_l,
            k=k,
            history_share=history_share,
            feedback_share=feedback_share,
            budgets=args.budgets,
            reps=args.reps,
            seed=args.seed + 1000 * idx,
        )
        split_frames.append(split_summary)
        raw_frames.append(raw)

    split_summary = pd.concat(split_frames, ignore_index=True)
    raw_summary = pd.concat(raw_frames, ignore_index=True)
    aggregate = aggregate_splits(split_summary)

    split_summary.to_csv(root / args.split_output, index=False)
    raw_summary.to_csv(root / args.raw_output, index=False)
    aggregate.to_csv(root / args.aggregate_output, index=False)

    display_cols = [
        "ratings_used",
        "split_configs",
        "best_cutoff_aware_reduction_vs_uniform_pct_mean",
        "best_cutoff_aware_reduction_vs_uniform_pct_min",
        "best_cutoff_aware_reduction_vs_uniform_pct_max",
        "share_splits_best_cutoff_aware_beats_uniform",
        "share_splits_cutoff_x_uncertainty_best",
    ]
    with pd.option_context("display.float_format", lambda x: f"{x:0.3f}"):
        print(aggregate[display_cols])


if __name__ == "__main__":
    main()
