"""Calibrated EIS illustration.

Replaces the purely illustrative EIS inputs with calibrated ones.

1. beta(S), empirical: fitted from MovieLens 100K. Ordered-probit cutpoints
   are fitted to the pooled rating distribution; item values w_i are
   recovered for the paper's 200-item catalog by matching observed mean
   ratings; kappa_bar(S) is the average ordered-probit Fisher information
   over the recovered item values. The binary scale uses the fitted 3|4
   cutpoint (like = rating >= 4). beta(5) = kappa_bar(5) / kappa_bar(2).

2. alpha(Q,S), anchored: the endpoints of the format menu are pinned to the
   published response-rate benchmarks already cited in the paper. The
   lightest format (binary, Q=3) is anchored at 0.29 (online surveys) and
   the heaviest (5-point, Q=7) at 0.13 (in-app surveys); completion
   declines log-linearly in questionnaire burden between the anchors.
   Burden is Q times a per-question effort for the richer scale. That
   effort is not separately identified by the cited benchmarks, so it is
   treated as a sensitivity parameter e5 in {1.0, 1.25, 1.5}.

   Two external checks discipline this construction. Liu and Wronski
   (2018, SSCR) report completion falling from 89% at 10 questions to
   79% at 40 across 25,080 real surveys, a log decline of about 0.004
   per question: within a 3-7 prompt menu, the question-count effect is
   mild, so the anchor-implied slope is conservative. Netflix's 2017
   in-product test reported that thumbs collected 200% more ratings
   than five-star prompts, a scale-burden ratio of about 3. The script
   therefore also computes an "in-product burden" scenario in which the
   5-point formats complete at one third the rate of the binary format
   of the same length.

Outputs eis_calibrated_summary.csv and prints the table rows.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np

ALPHA_LIGHT = 0.29  # online-survey benchmark (lightest menu format)
ALPHA_HEAVY = 0.13  # in-app benchmark (heaviest menu format)
EFFORT_GRID = [1.0, 1.25, 1.5]
FORMATS = [
    ("Binary", 2, 3),
    ("Binary", 2, 5),
    ("5-point", 5, 3),
    ("5-point", 5, 5),
    ("5-point", 5, 7),
]


def phi(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def Phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def fisher_info(w: float, thresholds: list[float]) -> float:
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


def expected_rating(w: float, cuts: list[float]) -> float:
    thresholds = [-np.inf] + cuts + [np.inf]
    value = 0.0
    for k, (lo, hi) in enumerate(zip(thresholds[:-1], thresholds[1:]), start=1):
        Phi_hi = 1.0 if np.isposinf(hi) else Phi(hi - w)
        Phi_lo = 0.0 if np.isneginf(lo) else Phi(lo - w)
        value += k * (Phi_hi - Phi_lo)
    return value


def solve_w(mean_rating: float, cuts: list[float]) -> float:
    lo, hi = -6.0, 6.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if expected_rating(mid, cuts) < mean_rating:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def calibrated_alpha(burden: float, burden_light: float, burden_heavy: float) -> float:
    decay = math.log(ALPHA_LIGHT / ALPHA_HEAVY) / (burden_heavy - burden_light)
    return ALPHA_LIGHT * math.exp(-decay * (burden - burden_light))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/ml-100k/u.data"),
        help="Path to MovieLens 100K u.data (tab-separated user, item, rating, ts).",
    )
    parser.add_argument(
        "--catalog-path",
        type=Path,
        default=Path("movielens_selected_catalog.csv"),
        help="Catalog CSV with item_id column (the paper's 200-item catalog).",
    )
    parser.add_argument("--out", type=Path, default=Path("eis_calibrated_summary.csv"))
    args = parser.parse_args()

    raw = np.loadtxt(args.data_path, dtype=int)
    items = raw[:, 1]
    ratings = raw[:, 2]

    # 1. Pooled ordered-probit cutpoints from the marginal rating distribution.
    shares = np.array([(ratings == k).mean() for k in range(1, 6)])
    cum = np.cumsum(shares)[:-1]
    cuts = [round(NormalDist().inv_cdf(float(c)), 4) for c in cum]

    # 2. Item values for the paper's catalog via mean-rating matching.
    catalog_ids = []
    with open(args.catalog_path) as f:
        for row in csv.DictReader(f):
            catalog_ids.append(int(row["item_id"]))
    w_values = np.array(
        [
            solve_w(float(ratings[items == item_id].mean()), cuts)
            for item_id in catalog_ids
            if np.any(items == item_id)
        ]
    )

    # 3. kappa_bar per scale and empirical beta.
    thresholds5 = [-np.inf] + cuts + [np.inf]
    thresholds2 = [-np.inf, cuts[2], np.inf]  # like = rating >= 4
    kappa5 = np.array([fisher_info(w, thresholds5) for w in w_values])
    kappa2 = np.array([fisher_info(w, thresholds2) for w in w_values])
    beta5 = float(kappa5.mean() / kappa2.mean())
    ratio = kappa5 / kappa2
    ratio_summary = (
        float(np.percentile(ratio, 10)),
        float(np.median(ratio)),
        float(np.percentile(ratio, 90)),
    )

    # 4. Calibrated completion and EIS per effort setting, plus the
    #    Netflix-ratio "in-product burden" scenario (5-point completes at
    #    one third of the same-length binary rate).
    NETFLIX_SCALE_RATIO = 3.0
    rows = []
    for effort5 in EFFORT_GRID:
        burden_light = 3.0  # binary, Q=3
        burden_heavy = 7.0 * effort5  # 5-point, Q=7
        for name, scale, questions in FORMATS:
            burden = questions * (1.0 if scale == 2 else effort5)
            alpha = calibrated_alpha(burden, burden_light, burden_heavy)
            alpha_binary_sameq = calibrated_alpha(
                float(questions), burden_light, burden_heavy
            )
            alpha_inproduct = (
                alpha_binary_sameq
                if scale == 2
                else alpha_binary_sameq / NETFLIX_SCALE_RATIO
            )
            beta = 1.0 if scale == 2 else beta5
            rows.append(
                {
                    "effort5": effort5,
                    "format": f"{name}, Q={questions}",
                    "burden_units": round(burden, 2),
                    "alpha_calibrated": round(alpha, 3),
                    "beta_empirical": round(beta, 3),
                    "eis": round(beta * questions * alpha, 3),
                    "alpha_inproduct": round(alpha_inproduct, 3),
                    "eis_inproduct": round(
                        beta * questions * alpha_inproduct, 3
                    ),
                }
            )

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        f.write(f"# rating_shares={[round(float(s), 4) for s in shares]}\n")
        f.write(f"# cutpoints_5point={cuts}\n")
        f.write(f"# binary_cutpoint(3|4)={cuts[2]}\n")
        f.write(f"# kappa_bar_5={kappa5.mean():.4f} kappa_bar_2={kappa2.mean():.4f}\n")
        f.write(f"# beta_5point_empirical={beta5:.4f}\n")
        f.write(
            f"# item_kappa_ratio_p10_median_p90={ratio_summary[0]:.3f},"
            f"{ratio_summary[1]:.3f},{ratio_summary[2]:.3f}\n"
        )
        f.write(f"# catalog_items_used={len(w_values)}\n")
        f.write(
            "# alpha anchors: binary Q=3 -> 0.29 (online-survey benchmark); "
            "5-point Q=7 -> 0.13 (in-app benchmark); log-linear in burden; "
            "5-point per-question effort e5 is a sensitivity parameter.\n"
        )

    print(f"rating shares (1..5): {[round(float(s), 4) for s in shares]}")
    print(f"fitted 5-point cutpoints: {cuts}")
    print(f"binary cutpoint (like = >=4): {cuts[2]}")
    print(
        f"catalog items used: {len(w_values)}; "
        f"w range [{w_values.min():.2f}, {w_values.max():.2f}]"
    )
    print(
        f"kappa_bar(5)={kappa5.mean():.4f}  kappa_bar(2)={kappa2.mean():.4f}  "
        f"beta(5)={beta5:.4f}"
    )
    print(f"item kappa ratio p10/median/p90: {ratio_summary}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
