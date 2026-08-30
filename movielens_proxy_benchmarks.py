#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import math
import subprocess
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
EXPECTED_ZIP_SHA256 = "50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229"
EXPECTED_UDATA_SHA256 = "06416e597f82b7342361e41163890c81036900f418ad91315590814211dca490"


@dataclass
class PolicyState:
    counts: np.ndarray
    sums: np.ndarray
    means: np.ndarray
    pointers: np.ndarray


@dataclass
class SplitInfo:
    data_file_sha256: str
    total_ratings: int
    selected_catalog_size: int
    catalog_selection_rule: str
    history_share: float
    feedback_share_of_post_history: float
    history_start_row: int
    history_end_row: int
    feedback_start_row: int
    feedback_end_row: int
    evaluation_start_row: int
    evaluation_end_row: int
    history_start_timestamp: int
    history_end_timestamp: int
    feedback_start_timestamp: int
    feedback_end_timestamp: int
    evaluation_start_timestamp: int
    evaluation_end_timestamp: int
    history_period_ratings: int
    feedback_period_ratings: int
    evaluation_period_ratings: int
    catalog_boundary_history_count: int
    catalog_boundary_tie_count: int
    catalog_boundary_selected_count: int
    catalog_history_ratings: int
    catalog_feedback_ratings: int
    catalog_evaluation_ratings: int
    min_feedback_ratings_per_item: int
    min_evaluation_ratings_per_item: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str, label: str) -> None:
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} checksum mismatch for {path}: expected {expected}, got {actual}."
        )


def download_file(url: str, output_path: Path) -> None:
    try:
        urllib.request.urlretrieve(url, output_path)
        return
    except urllib.error.URLError as first_error:
        try:
            subprocess.run(
                ["curl", "--fail", "--location", "--silent", "--show-error", url, "--output", str(output_path)],
                check=True,
            )
            return
        except (FileNotFoundError, subprocess.CalledProcessError) as second_error:
            raise RuntimeError(
                f"Could not download {url}. Python urllib failed with {first_error}; "
                f"curl fallback failed with {second_error}."
            ) from first_error


def ensure_dataset(root: Path, skip_download: bool = False) -> Path:
    data_dir = root / "data"
    zip_path = data_dir / "ml-100k.zip"
    data_path = data_dir / "ml-100k" / "u.data"
    if data_path.exists():
        verify_sha256(data_path, EXPECTED_UDATA_SHA256, "MovieLens u.data")
        return data_path

    if skip_download:
        raise FileNotFoundError(
            f"MovieLens u.data not found at {data_path}. Provide --data-path "
            "or rerun without --skip-download to fetch the official GroupLens archive."
        )

    data_dir.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        print(f"Downloading MovieLens 100K to {zip_path}...")
        download_file(DATA_URL, zip_path)
    verify_sha256(zip_path, EXPECTED_ZIP_SHA256, "MovieLens ml-100k.zip")

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)
    verify_sha256(data_path, EXPECTED_UDATA_SHA256, "MovieLens u.data")
    return data_path


def resolve_data_path(root: Path, data_path: Path | None, skip_download: bool = False) -> Path:
    if data_path is None:
        return ensure_dataset(root, skip_download=skip_download)
    verify_sha256(data_path, EXPECTED_UDATA_SHA256, "MovieLens u.data")
    return data_path


def period_min_timestamp(frame: pd.DataFrame) -> int:
    return int(frame["timestamp"].min()) if not frame.empty else -1


def period_max_timestamp(frame: pd.DataFrame) -> int:
    return int(frame["timestamp"].max()) if not frame.empty else -1


def prepare_dataset(
    data_path: Path, top_l: int, history_share: float, feedback_share: float
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray, SplitInfo, pd.DataFrame]:
    if not 0.0 < history_share < 1.0:
        raise ValueError("--history-share must be between 0 and 1.")
    if not 0.0 < feedback_share < 1.0:
        raise ValueError("--feedback-share must be between 0 and 1.")

    cols = ["user_id", "item_id", "rating", "timestamp"]
    df = pd.read_csv(data_path, sep="\t", names=cols, engine="python")
    df = df.sort_values("timestamp").reset_index(drop=True)

    history_end = int(len(df) * history_share)
    feedback_end = history_end + int((len(df) - history_end) * feedback_share)
    history_all = df.iloc[:history_end].copy()
    feedback_all = df.iloc[history_end:feedback_end].copy()
    evaluation_all = df.iloc[feedback_end:].copy()

    history_item_counts = (
        history_all.groupby("item_id")
        .size()
        .rename("history_count")
        .reset_index()
    )
    ranked_history_items = history_item_counts.sort_values(
        ["history_count", "item_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    top_catalog = ranked_history_items.head(top_l).copy()
    top_catalog["catalog_rank"] = np.arange(1, len(top_catalog) + 1)
    top_items = top_catalog["item_id"].astype(int).tolist()
    if len(top_items) < top_l:
        raise ValueError(
            f"History period contains only {len(top_items)} distinct items; "
            f"cannot build a catalog of L={top_l}."
        )
    boundary_count = int(top_catalog["history_count"].iloc[-1])
    boundary_tie_count = int((history_item_counts["history_count"] == boundary_count).sum())
    boundary_selected_count = int((top_catalog["history_count"] == boundary_count).sum())

    item_map = {item_id: idx for idx, item_id in enumerate(top_items)}
    history = history_all[history_all["item_id"].isin(top_items)].copy()
    feedback = feedback_all[feedback_all["item_id"].isin(top_items)].copy()
    evaluation = evaluation_all[evaluation_all["item_id"].isin(top_items)].copy()
    for frame in (history, feedback, evaluation):
        frame["item_idx"] = frame["item_id"].map(item_map)

    history_counts = history.groupby("item_idx").size().reindex(range(top_l), fill_value=0).to_numpy(dtype=np.int64)
    history_sums = history.groupby("item_idx")["rating"].sum().reindex(range(top_l), fill_value=0.0).to_numpy(dtype=float)
    feedback_ratings = [
        feedback.loc[feedback["item_idx"] == idx, "rating"].to_numpy(dtype=float)
        for idx in range(top_l)
    ]
    evaluation_ratings = [
        evaluation.loc[evaluation["item_idx"] == idx, "rating"].to_numpy(dtype=float)
        for idx in range(top_l)
    ]

    feedback_lengths = np.array([len(ratings) for ratings in feedback_ratings], dtype=np.int64)
    evaluation_lengths = np.array([len(ratings) for ratings in evaluation_ratings], dtype=np.int64)
    if np.any(feedback_lengths == 0) or np.any(evaluation_lengths == 0):
        raise ValueError(
            "History-selected catalog contains items with no feedback-pool or "
            "evaluation-holdout ratings. Reduce --top-l or adjust the split."
        )

    oracle_means = np.array([ratings.mean() for ratings in evaluation_ratings], dtype=float)

    global_history_mean = float(history["rating"].mean())
    zero_history = history_counts == 0
    if zero_history.any():
        history_counts = history_counts.copy()
        history_sums = history_sums.copy()
        history_counts[zero_history] = 1
        history_sums[zero_history] = global_history_mean

    feedback_count_by_item = feedback_all.groupby("item_id").size()
    evaluation_count_by_item = evaluation_all.groupby("item_id").size()
    selected_catalog = top_catalog[
        ["catalog_rank", "item_id", "history_count"]
    ].copy()
    selected_catalog["feedback_count"] = (
        selected_catalog["item_id"].map(feedback_count_by_item).fillna(0).astype(int)
    )
    selected_catalog["evaluation_count"] = (
        selected_catalog["item_id"].map(evaluation_count_by_item).fillna(0).astype(int)
    )

    split_info = SplitInfo(
        data_file_sha256=file_sha256(data_path),
        total_ratings=len(df),
        selected_catalog_size=top_l,
        catalog_selection_rule="history_count_desc_then_item_id_asc",
        history_share=history_share,
        feedback_share_of_post_history=feedback_share,
        history_start_row=0,
        history_end_row=history_end - 1,
        feedback_start_row=history_end,
        feedback_end_row=feedback_end - 1,
        evaluation_start_row=feedback_end,
        evaluation_end_row=len(df) - 1,
        history_start_timestamp=period_min_timestamp(history_all),
        history_end_timestamp=period_max_timestamp(history_all),
        feedback_start_timestamp=period_min_timestamp(feedback_all),
        feedback_end_timestamp=period_max_timestamp(feedback_all),
        evaluation_start_timestamp=period_min_timestamp(evaluation_all),
        evaluation_end_timestamp=period_max_timestamp(evaluation_all),
        history_period_ratings=len(history_all),
        feedback_period_ratings=len(feedback_all),
        evaluation_period_ratings=len(evaluation_all),
        catalog_boundary_history_count=boundary_count,
        catalog_boundary_tie_count=boundary_tie_count,
        catalog_boundary_selected_count=boundary_selected_count,
        catalog_history_ratings=len(history),
        catalog_feedback_ratings=len(feedback),
        catalog_evaluation_ratings=len(evaluation),
        min_feedback_ratings_per_item=int(feedback_lengths.min()),
        min_evaluation_ratings_per_item=int(evaluation_lengths.min()),
    )

    return history_counts, history_sums, feedback_ratings, oracle_means, split_info, selected_catalog


def choose_random(rng: np.random.Generator, eligible: np.ndarray) -> int:
    return int(rng.choice(eligible))


def boundary_weight(means: np.ndarray, k: int) -> np.ndarray:
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
    return np.exp(-((diff / scale) ** 2))


def uncertainty_drop(counts: np.ndarray) -> np.ndarray:
    counts = counts.astype(float)
    return 1.0 / counts - 1.0 / (counts + 1.0)


def pick_argmax_with_random_tie(
    scores: np.ndarray, eligible: np.ndarray, rng: np.random.Generator
) -> int:
    eligible_scores = scores[eligible]
    max_score = eligible_scores.max()
    best = eligible[np.isclose(eligible_scores, max_score)]
    return int(rng.choice(best))


def select_item(policy: str, state: PolicyState, eligible: np.ndarray, k: int, rng: np.random.Generator) -> int:
    if policy == "uniform":
        return choose_random(rng, eligible)

    drops = uncertainty_drop(state.counts)
    if policy == "uncertainty_only":
        return pick_argmax_with_random_tie(drops, eligible, rng)

    boundary = boundary_weight(state.means, k)
    if policy == "cutoff_only":
        return pick_argmax_with_random_tie(boundary, eligible, rng)

    if policy == "cutoff_x_uncertainty":
        return pick_argmax_with_random_tie(boundary * drops, eligible, rng)

    raise ValueError(f"Unknown policy: {policy}")


def topk_gap(estimates: np.ndarray, oracle_means: np.ndarray, k: int, oracle_top_sum: float) -> float:
    chosen = np.argpartition(estimates, -k)[-k:]
    return float(oracle_top_sum - oracle_means[chosen].sum())


def run_policy(
    policy: str,
    history_counts: np.ndarray,
    history_sums: np.ndarray,
    future_orders: list[np.ndarray],
    oracle_means: np.ndarray,
    budgets: list[int],
    k: int,
    rng: np.random.Generator,
) -> list[float]:
    counts = history_counts.astype(np.int64).copy()
    sums = history_sums.astype(float).copy()
    means = sums / counts
    pointers = np.zeros(len(future_orders), dtype=np.int64)
    state = PolicyState(counts=counts, sums=sums, means=means, pointers=pointers)

    oracle_top_sum = float(np.sort(oracle_means)[-k:].sum())
    budget_set = set(budgets)
    max_budget = max(budgets)
    results = {0: topk_gap(state.means, oracle_means, k, oracle_top_sum)}

    for step in range(1, max_budget + 1):
        eligible = np.flatnonzero(state.pointers < np.array([len(arr) for arr in future_orders]))
        if len(eligible) == 0:
            results[step] = topk_gap(state.means, oracle_means, k, oracle_top_sum)
            continue

        choice = select_item(policy, state, eligible, k, rng)
        rating = future_orders[choice][state.pointers[choice]]
        state.pointers[choice] += 1
        state.counts[choice] += 1
        state.sums[choice] += rating
        state.means[choice] = state.sums[choice] / state.counts[choice]

        if step in budget_set:
            results[step] = topk_gap(state.means, oracle_means, k, oracle_top_sum)

    return [results[budget] for budget in budgets]


def build_future_orders(
    future_ratings: list[np.ndarray], rng: np.random.Generator
) -> list[np.ndarray]:
    orders = []
    for ratings in future_ratings:
        shuffled = ratings.copy()
        rng.shuffle(shuffled)
        orders.append(shuffled)
    return orders


def summarize_runs(records: dict[str, list[list[float]]], budgets: list[int]) -> pd.DataFrame:
    summary = pd.DataFrame({"ratings_used": budgets})
    arrays: dict[str, np.ndarray] = {}
    for policy, runs in records.items():
        arr = np.array(runs, dtype=float)
        arrays[policy] = arr
        summary[f"{policy}_mean"] = arr.mean(axis=0)
        summary[f"{policy}_sd"] = arr.std(axis=0, ddof=1)
    baseline = summary["uniform_mean"]
    summary["cutoff_only_improvement_vs_uniform_%"] = 100.0 * (
        baseline - summary["cutoff_only_mean"]
    ) / baseline.clip(lower=1e-8)
    summary["improvement_vs_uniform_%"] = 100.0 * (
        baseline - summary["cutoff_x_uncertainty_mean"]
    ) / baseline.clip(lower=1e-8)
    summary["improvement_vs_cutoff_%"] = 100.0 * (
        summary["cutoff_only_mean"] - summary["cutoff_x_uncertainty_mean"]
    ) / summary["cutoff_only_mean"].clip(lower=1e-8)
    summary["best_cutoff_aware_mean"] = np.minimum(
        summary["cutoff_only_mean"], summary["cutoff_x_uncertainty_mean"]
    )
    summary["best_cutoff_aware_improvement_vs_uniform_%"] = 100.0 * (
        baseline - summary["best_cutoff_aware_mean"]
    ) / baseline.clip(lower=1e-8)

    def add_paired_columns(policy: str, baseline_policy: str, label: str) -> None:
        improvement = arrays[baseline_policy] - arrays[policy]
        reps = improvement.shape[0]
        if reps > 1:
            se = improvement.std(axis=0, ddof=1) / math.sqrt(reps)
        else:
            se = np.zeros(improvement.shape[1], dtype=float)
        summary[f"{label}_paired_improvement_mean"] = improvement.mean(axis=0)
        summary[f"{label}_paired_improvement_se"] = se
        summary[f"{label}_win_rate"] = (improvement > 0.0).mean(axis=0)

    for policy in ("uncertainty_only", "cutoff_only", "cutoff_x_uncertainty"):
        add_paired_columns(policy, "uniform", f"{policy}_vs_uniform")
    add_paired_columns(
        "cutoff_x_uncertainty",
        "cutoff_only",
        "cutoff_x_uncertainty_vs_cutoff_only",
    )
    return summary


def plot_results(summary: pd.DataFrame, output_path: Path, reps: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "uniform": "#7f7f7f",
        "uncertainty_only": "#1f77b4",
        "cutoff_only": "#ff7f0e",
        "cutoff_x_uncertainty": "#2ca02c",
    }
    labels = {
        "uniform": "Uniform",
        "uncertainty_only": "Uncertainty-only",
        "cutoff_only": "Cutoff-only",
        "cutoff_x_uncertainty": "Cutoff x uncertainty",
    }

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = summary["ratings_used"].to_numpy()
    for policy in labels:
        mean = summary[f"{policy}_mean"].to_numpy()
        sd = summary[f"{policy}_sd"].to_numpy()
        ci95 = 1.96 * sd / math.sqrt(reps)
        ax.plot(x, mean, marker="o", linewidth=2.2, color=colors[policy], label=labels[policy])
        ax.fill_between(x, mean - ci95, mean + ci95, color=colors[policy], alpha=0.12)

    ax.set_xlabel("Additional ratings collected (budget M)")
    ax.set_ylabel("Top-K value gap")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
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
    parser.add_argument("--top-l", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--history-share", type=float, default=0.2)
    parser.add_argument(
        "--feedback-share",
        type=float,
        default=0.5,
        help="Fraction of post-history ratings used as the feedback pool; the rest is evaluation holdout.",
    )
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260327)
    parser.add_argument(
        "--summary-output",
        "--output",
        dest="summary_output",
        default="movielens_experiment_summary.csv",
        help="CSV path for the summary table.",
    )
    parser.add_argument(
        "--plot-output",
        "--plot",
        dest="plot_output",
        default="movielens_gap_vs_budget.png",
        help="PNG path for the budget-gap plot.",
    )
    parser.add_argument(
        "--raw-output",
        default="movielens_experiment_raw.csv",
        help="CSV path for one row per repetition, policy, and budget.",
    )
    parser.add_argument(
        "--split-output",
        default="movielens_split_manifest.csv",
        help="CSV path for the chronological split manifest.",
    )
    parser.add_argument(
        "--catalog-output",
        default="movielens_selected_catalog.csv",
        help="CSV path for the deterministic history-selected catalog.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip plotting and only write the summary CSV.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not download MovieLens data; fail if --data-path or data/ml-100k/u.data is unavailable.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify the MovieLens checksum and deterministic clean split, then exit without running policies or writing outputs.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    data_path = resolve_data_path(root, args.data_path, skip_download=args.skip_download)
    (
        history_counts,
        history_sums,
        feedback_ratings,
        oracle_means,
        split_info,
        selected_catalog,
    ) = prepare_dataset(
        data_path,
        top_l=args.top_l,
        history_share=args.history_share,
        feedback_share=args.feedback_share,
    )

    if args.verify_only:
        print(f"Verified MovieLens data: {data_path}")
        print(f"u.data SHA256: {split_info.data_file_sha256}")
        print(
            "Clean split verified: "
            f"L={len(oracle_means)}, history={split_info.catalog_history_ratings}, "
            f"feedback={split_info.catalog_feedback_ratings}, "
            f"evaluation={split_info.catalog_evaluation_ratings}, "
            f"min feedback/evaluation per item="
            f"{split_info.min_feedback_ratings_per_item}/"
            f"{split_info.min_evaluation_ratings_per_item}."
        )
        print(
            "Catalog tie-break: history count descending, item ID ascending; "
            f"boundary count={split_info.catalog_boundary_history_count}, "
            f"selected boundary ties={split_info.catalog_boundary_selected_count}/"
            f"{split_info.catalog_boundary_tie_count}."
        )
        return

    budgets = list(range(0, 2001, 200))
    policies = ["uniform", "uncertainty_only", "cutoff_only", "cutoff_x_uncertainty"]
    records: dict[str, list[list[float]]] = {policy: [] for policy in policies}
    raw_rows = []

    print(
        f"Using L={len(oracle_means)}, K={args.k}, repetitions={args.reps}, "
        f"history={split_info.catalog_history_ratings}, "
        f"feedback_pool={split_info.catalog_feedback_ratings}, "
        f"evaluation_holdout={split_info.catalog_evaluation_ratings}"
    )
    master_rng = np.random.default_rng(args.seed)

    for rep in range(args.reps):
        rep_seed = int(master_rng.integers(0, 2**32 - 1))
        future_orders = build_future_orders(feedback_ratings, np.random.default_rng(rep_seed))
        for offset, policy in enumerate(policies):
            policy_rng = np.random.default_rng(rep_seed + 1000 * (offset + 1))
            gaps = run_policy(
                policy=policy,
                history_counts=history_counts,
                history_sums=history_sums,
                future_orders=future_orders,
                oracle_means=oracle_means,
                budgets=budgets,
                k=args.k,
                rng=policy_rng,
            )
            records[policy].append(gaps)
            for budget, gap in zip(budgets, gaps):
                raw_rows.append(
                    {
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
    summary.to_csv(root / args.summary_output, index=False)
    pd.DataFrame([asdict(split_info)]).to_csv(root / args.split_output, index=False)
    selected_catalog.to_csv(root / args.catalog_output, index=False)
    pd.DataFrame(raw_rows).to_csv(root / args.raw_output, index=False)
    if not args.skip_plot:
        plot_results(summary, root / args.plot_output, reps=args.reps)

    display_cols = [
        "ratings_used",
        "uniform_mean",
        "uncertainty_only_mean",
        "cutoff_only_mean",
        "cutoff_x_uncertainty_mean",
        "improvement_vs_uniform_%",
    ]
    with pd.option_context("display.float_format", lambda x: f"{x:0.3f}"):
        print(summary[display_cols])


if __name__ == "__main__":
    main()
