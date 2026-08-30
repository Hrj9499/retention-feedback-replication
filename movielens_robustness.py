#!/usr/bin/env python3

from __future__ import annotations

import argparse
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
    ("Baseline", 200, 10, 0.20, 0.50),
    ("Smaller catalog", 150, 10, 0.20, 0.50),
    ("Smaller box", 200, 8, 0.20, 0.50),
    ("Deeper history", 200, 10, 0.30, 0.50),
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
    parser.add_argument("--seed", type=int, default=20260416)
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[1000, 2000],
        help="Budgets to summarize in the robustness table.",
    )
    parser.add_argument(
        "--output",
        default="movielens_robustness_summary.csv",
        help="CSV path for the robustness summary.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not download MovieLens data; fail if --data-path or data/ml-100k/u.data is unavailable.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify the MovieLens checksum and robustness clean splits, then exit without running policies or writing outputs.",
    )
    return parser.parse_args()


def run_config(
    root: Path,
    data_path: Path,
    label: str,
    top_l: int,
    k: int,
    history_share: float,
    feedback_share: float,
    budgets: list[int],
    reps: int,
    seed: int,
) -> pd.DataFrame:
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

    summary = summarize_runs(records, budgets)
    summary["config"] = label
    summary["top_l"] = top_l
    summary["k"] = k
    summary["history_share"] = history_share
    summary["feedback_share"] = feedback_share
    summary["catalog_history_ratings"] = split_info.catalog_history_ratings
    summary["catalog_feedback_ratings"] = split_info.catalog_feedback_ratings
    summary["catalog_evaluation_ratings"] = split_info.catalog_evaluation_ratings
    cxu_mean = summary["cutoff_x_uncertainty_mean"]
    cutoff_mean = summary["cutoff_only_mean"]
    summary["best_policy"] = np.select(
        [
            np.isclose(cxu_mean, cutoff_mean, atol=1e-10),
            cxu_mean < cutoff_mean,
        ],
        ["Tie", "cutoff x uncertainty"],
        default="cutoff-only",
    )
    return summary[
        [
            "config",
            "top_l",
            "k",
            "history_share",
            "feedback_share",
            "ratings_used",
            "uniform_mean",
            "cutoff_only_mean",
            "cutoff_x_uncertainty_mean",
            "best_cutoff_aware_mean",
            "best_cutoff_aware_improvement_vs_uniform_%",
            "improvement_vs_uniform_%",
            "improvement_vs_cutoff_%",
            "best_policy",
        ]
    ]


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
                f"history={split_info.catalog_history_ratings}, "
                f"feedback={split_info.catalog_feedback_ratings}, "
                f"evaluation={split_info.catalog_evaluation_ratings}, "
                f"min feedback/evaluation="
                f"{split_info.min_feedback_ratings_per_item}/"
                f"{split_info.min_evaluation_ratings_per_item}."
            )
        return

    frames = []
    for idx, (label, top_l, k, history_share, feedback_share) in enumerate(CONFIGS):
        frames.append(
            run_config(
                root=root,
                data_path=data_path,
                label=label,
                top_l=top_l,
                k=k,
                history_share=history_share,
                feedback_share=feedback_share,
                budgets=args.budgets,
                reps=args.reps,
                seed=args.seed + 100 * idx,
            )
        )

    summary = pd.concat(frames, ignore_index=True)
    summary.to_csv(root / args.output, index=False)

    with pd.option_context("display.float_format", lambda x: f"{x:0.3f}"):
        print(summary)


if __name__ == "__main__":
    main()
