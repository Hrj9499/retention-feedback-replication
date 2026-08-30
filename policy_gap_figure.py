#!/usr/bin/env python3
"""Render the main-allocation policy comparison figure (Section 7.2).

Reads a dense-checkpoint summary from synthetic_allocation_benchmarks.py
(budget multiples 0,2,5,...,100) and plots mean next-box value gaps with
one-standard-error bands, one panel per prior. Deterministic; no simulation.
"""

from __future__ import annotations

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = [
    # (csv stem, label, hex, marker, linewidth, zorder)
    ("cutoff", "Guarded cutoff", "#2a78d6", "o", 2.6, 6),
    ("even", "Even", "#eb6834", "s", 1.5, 3),
    ("uncertainty", "Uncertainty-only", "#1baf7a", "^", 1.5, 3),
    ("combined", "Combined cutoff $\\times$ uncertainty", "#eda100", "D", 1.5, 4),
    ("local_marginal", "Estimated marginal value", "#e87ba4", "v", 1.5, 3),
]
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="synthetic_allocation_dense.csv")
    parser.add_argument("--output", default="policy_gap_comparison.png")
    parser.add_argument("--catalog-size", type=int, default=60)
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.input)))
    priors = []
    for r in rows:
        if r["prior"] not in priors:
            priors.append(r["prior"])

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True)
    titles = {priors[0]: "AR(1) prior ($\\rho=0.6$)", priors[1]: "Independent prior"}
    for ax, prior in zip(axes, priors):
        sub = sorted(
            (r for r in rows if r["prior"] == prior), key=lambda r: int(r["budget"])
        )
        budgets = [int(r["budget"]) / args.catalog_size for r in sub]
        for stem, label, hexc, marker, lw, z in SERIES:
            mean = [float(r[f"{stem}_mean"]) for r in sub]
            se = [float(r[f"{stem}_se"]) for r in sub]
            lo = [m - s for m, s in zip(mean, se)]
            hi = [m + s for m, s in zip(mean, se)]
            ax.fill_between(budgets, lo, hi, color=hexc, alpha=0.16, linewidth=0, zorder=z - 1)
            ax.plot(
                budgets, mean, marker=marker, color=hexc, linewidth=lw,
                markersize=4.5, zorder=z, label=label,
                markeredgecolor="white", markeredgewidth=0.8,
            )
        ax.set_yscale("log")
        ax.set_xlim(-2, 103)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_xticklabels(["0", "$20L$", "$40L$", "$60L$", "$80L$", "$100L$"])
        ax.set_title(titles[prior], fontsize=10, color=TEXT_PRIMARY)
        ax.set_xlabel("Additional response budget", fontsize=9, color=TEXT_SECONDARY)
        ax.grid(True, which="major", axis="y", color="#d8d7d2", linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#a9a8a2")
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8.5)
    axes[0].set_ylabel("Mean next-box value gap", fontsize=9, color=TEXT_SECONDARY)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
        ncol=5, frameon=False, fontsize=8, columnspacing=1.1,
        handletextpad=0.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(args.output, dpi=300, facecolor="white")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
