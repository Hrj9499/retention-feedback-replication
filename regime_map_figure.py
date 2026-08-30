#!/usr/bin/env python3
"""Render the refinement boundary map (Section 7.3).

Reads combined_vs_cutoff_regime_map_summary.csv and draws a heatmap of
the mean improvement of the combined rule over the guarded cutoff rule,
by budget and regime slice. Deterministic; no simulation.
"""

from __future__ import annotations

import argparse
import csv
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
# diverging pair: blue (refinement helps) <-> red (refinement hurts), gray midpoint
CMAP = LinearSegmentedColormap.from_list(
    "div", ["#e34948", "#f0efec", "#2a78d6"]
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="combined_vs_cutoff_regime_map_summary.csv")
    parser.add_argument("--output", default="refinement_boundary_map.png")
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.input)))
    budgets = ["1", "3", "10", "30"]
    slices = [
        ("All regimes", lambda r: True),
        ("Heterogeneous\ninformation", lambda r: r["information_stock"] == "heterogeneous"),
        ("Crowded cutoff", lambda r: r["density_scenario"] == "crowded"),
    ]
    grid = []
    for _, pred in slices:
        grid.append([
            statistics.mean(
                float(r["combined_improvement_vs_cutoff_mean"])
                for r in rows
                if r["budget_multiple"] == b and pred(r)
            )
            for b in budgets
        ])

    vmax = max(abs(v) for row in grid for v in row)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    ax.imshow(grid, cmap=CMAP, norm=norm, aspect="auto")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["$L$", "$3L$", "$10L$", "$30L$"], fontsize=9, color=TEXT_SECONDARY)
    ax.set_yticks(range(3))
    ax.set_yticklabels([name for name, _ in slices], fontsize=9, color=TEXT_SECONDARY)
    ax.set_xlabel("Additional response budget", fontsize=9, color=TEXT_SECONDARY)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    # 2px white gaps between cells
    for x in range(1, 4):
        ax.axvline(x - 0.5, color="white", linewidth=2)
    for y in range(1, 3):
        ax.axhline(y - 0.5, color="white", linewidth=2)
    for yi, row in enumerate(grid):
        for xi, v in enumerate(row):
            ax.text(
                xi, yi, f"{v:+.3f}", ha="center", va="center",
                fontsize=9, color=TEXT_PRIMARY,
                fontweight="bold" if v > 0 else "normal",
            )
    ax.set_title(
        "Gain from adding the uncertainty term to the guarded cutoff rule",
        fontsize=9.5, color=TEXT_PRIMARY, pad=10,
    )
    fig.tight_layout()
    fig.savefig(args.output, dpi=300, facecolor="white")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
