#!/usr/bin/env python3
"""Render the main-allocation policy comparison figure (Figure 2).

Reads synthetic_allocation_summary.csv and plots mean next-box value gaps
by policy and budget, one panel per prior. Deterministic; no simulation.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = [
    # (csv column, label, hex, marker, linewidth, zorder)
    ("cutoff_mean", "Guarded cutoff", "#2a78d6", "o", 2.8, 5),
    ("even_mean", "Even", "#eb6834", "s", 1.4, 3),
    ("uncertainty_mean", "Uncertainty-only", "#1baf7a", "^", 1.4, 3),
    ("combined_mean", "Combined cutoff $\\times$ uncertainty", "#eda100", "D", 1.4, 3),
    ("local_marginal_mean", "Estimated marginal value", "#e87ba4", "v", 1.4, 3),
]
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="synthetic_allocation_summary.csv")
    parser.add_argument("--output", default="policy_gap_comparison.png")
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.input)))
    priors = []
    for r in rows:
        if r["prior"] not in priors:
            priors.append(r["prior"])

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    titles = {priors[0]: "AR(1) prior ($\\rho=0.6$)", priors[1]: "Independent prior"}
    for ax, prior in zip(axes, priors):
        sub = [r for r in rows if r["prior"] == prior]
        budgets = [int(r["budget"]) / 60 for r in sub]
        for col, label, hexc, marker, lw, z in SERIES:
            vals = [float(r[col]) for r in sub]
            ax.plot(
                budgets, vals, marker=marker, color=hexc, linewidth=lw,
                markersize=6.5, zorder=z, label=label,
                markeredgecolor="white", markeredgewidth=1.0,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([10, 30, 100])
        ax.set_xticklabels(["$10L$", "$30L$", "$100L$"])
        ax.minorticks_off()
        ax.set_title(titles[prior], fontsize=10, color=TEXT_PRIMARY)
        ax.set_xlabel("Response budget", fontsize=9, color=TEXT_SECONDARY)
        ax.grid(True, which="major", axis="y", color="#d8d7d2", linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#a9a8a2")
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8.5)
    axes[0].set_ylabel("Mean next-box value gap", fontsize=9, color=TEXT_SECONDARY)

    # direct label for the protagonist on the right panel
    sub = [r for r in rows if r["prior"] == priors[1]]
    gx = int(sub[-1]["budget"]) / 60
    gy = float(sub[-1]["cutoff_mean"])
    axes[1].annotate(
        "Guarded cutoff", xy=(gx, gy), xytext=(-10, 10),
        textcoords="offset points", ha="right", fontsize=8.5,
        color=TEXT_PRIMARY, fontweight="bold",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
        ncol=5, frameon=False, fontsize=8, columnspacing=1.2,
        handletextpad=0.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(args.output, dpi=300, facecolor="white")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
