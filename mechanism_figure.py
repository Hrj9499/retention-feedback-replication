"""Generate the cutoff-mechanism illustration (Figure 1).

One synthetic catalog, items sorted by current estimated value, with
uncertainty intervals. The shipment cutoff splits the catalog into
safely-in, contested, and safely-out groups; the contested band is where
answers have decision value. Deterministic given the seed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--L", type=int, default=30)
    parser.add_argument("--K", type=int, default=6)
    parser.add_argument("--out", type=Path, default=Path("cutoff_mechanism.png"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    L, K = args.L, args.K
    means = np.sort(rng.normal(0.0, 1.0, L))[::-1]
    # Heterogeneous uncertainty; a bit larger near the middle of the pack.
    sds = 0.10 + 0.12 * rng.random(L)

    cutoff = 0.5 * (means[K - 1] + means[K])
    lo = means - 2 * sds
    hi = means + 2 * sds
    contested = (lo < cutoff) & (hi > cutoff)
    safely_in = (~contested) & (means > cutoff)

    color_in = "#2563EB"      # blue
    color_contested = "#D97706"  # amber
    color_out = "#9CA3AF"     # gray
    colors = np.where(contested, color_contested,
                      np.where(safely_in, color_in, color_out))

    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=200)
    x = np.arange(1, L + 1)
    for i in range(L):
        ax.plot([x[i], x[i]], [lo[i], hi[i]], color=colors[i],
                lw=1.6, solid_capstyle="round", zorder=2,
                alpha=0.9 if contested[i] else 0.65)
    ax.scatter(x[safely_in], means[safely_in], s=26, color=color_in,
               zorder=3, label=None)
    ax.scatter(x[contested], means[contested], s=34, color=color_contested,
               marker="D", zorder=3)
    out_mask = ~(safely_in | contested)
    ax.scatter(x[out_mask], means[out_mask], s=26, facecolors="white",
               edgecolors=color_out, linewidths=1.4, zorder=3)

    ax.hlines(cutoff, 0.2, L + 0.8, color="#374151", lw=1.2,
              linestyles=(0, (5, 3)), zorder=1)
    ax.text(L + 1.2, cutoff + 0.05, "shipment cutoff\n(top-$K$ boundary)",
            va="center", ha="left", fontsize=9, color="#374151")

    # Group annotations.
    first_contested = int(np.argmax(contested)) + 1
    last_contested = int(L - np.argmax(contested[::-1]))
    ax.annotate("safely in:\nmore answers change nothing",
                xy=(max(1.5, (first_contested - 1) / 2), means[1] + 0.15),
                fontsize=9, color=color_in, ha="left")
    ax.annotate("contested:\nanswers can still change the box",
                xy=((first_contested + last_contested) / 2, min(lo[contested]) - 0.32),
                fontsize=9, color="#92400E", ha="center")
    ax.annotate("safely out",
                xy=(L - 3.4, means[-2] - 0.42), fontsize=9, color="#6B7280",
                ha="left")

    ax.set_xlabel("items, sorted by current estimated value", fontsize=9)
    ax.set_ylabel("estimated value (interval: $\\pm 2$ sd)", fontsize=9)
    ax.set_xlim(0.2, L + 5.6)
    ax.tick_params(labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#9CA3AF")
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
