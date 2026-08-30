"""Low-count surrogate audit (new, self-contained).

The archived surrogate-fidelity audit starts at ten responses per item.
This audit extends the comparison to one, three, and five responses per
item, the warm-start regime of the experiments, with a fully stated and
reproducible methodology.

Coupled design. For each catalog replication, true values w_i are drawn
from the prior, and each item receives N latent experiences
u_t = w_i + eps_t with the same draws feeding both pipelines. Pipeline
A (exact ordinal): ratings R_t = bin(u_t) under the ordered-probit
thresholds; the posterior mean of w_i is computed by data-augmentation
Gibbs sampling. Pipeline B (Gaussian surrogate): the same shocks are
scaled to the information-equivalent signal y_t = w_i +
(u_t - w_i)/sqrt(kappa(S)), and the posterior mean is the conjugate
Gaussian update. The audit reports the agreement of the two posterior
mean vectors and of the induced top-K sets.

Outputs surrogate_lowcount_summary.csv.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.stats import truncnorm

THRESHOLDS = {
    2: np.array([-np.inf, 0.0, np.inf]),
    5: np.array([-np.inf, -1.0, -0.2, 0.2, 1.0, np.inf]),
}


def phi(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def Phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def fisher_info(w: float, scale: int) -> float:
    th = THRESHOLDS[scale]
    total = 0.0
    for lo, hi in zip(th[:-1], th[1:]):
        Phi_hi = 1.0 if np.isposinf(hi) else Phi(hi - w)
        Phi_lo = 0.0 if np.isneginf(lo) else Phi(lo - w)
        prob = max(Phi_hi - Phi_lo, 1e-12)
        d_hi = 0.0 if np.isposinf(hi) else phi(hi - w)
        d_lo = 0.0 if np.isneginf(lo) else phi(lo - w)
        total += (d_lo - d_hi) ** 2 / prob
    return total


def expected_kappa(scale: int, gh_x, gh_w) -> float:
    terms = [
        wgt * fisher_info(math.sqrt(2.0) * x, scale)
        for x, wgt in zip(gh_x, gh_w)
    ]
    return sum(terms) / math.sqrt(math.pi)


def gibbs_posterior_means(ratings, scale, iters, burn, rng):
    """Per-item data-augmentation Gibbs for w with N(0,1) prior.

    ratings: (L, N) integer matrix of ordinal responses (1-based).
    Returns the posterior-mean estimate per item.
    """
    L, N = ratings.shape
    th = THRESHOLDS[scale]
    lo_bounds = th[ratings - 1]
    hi_bounds = th[ratings]
    w = np.zeros(L)
    total = np.zeros(L)
    kept = 0
    for it in range(iters):
        a = lo_bounds - w[:, None]
        b = hi_bounds - w[:, None]
        u = truncnorm.rvs(a, b, loc=w[:, None], scale=1.0, random_state=rng)
        post_prec = 1.0 + N
        post_mean = u.sum(axis=1) / post_prec
        w = post_mean + rng.standard_normal(L) / math.sqrt(post_prec)
        if it >= burn:
            total += post_mean
            kept += 1
    return total / kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--L", type=int, default=60)
    parser.add_argument("--K", type=int, default=6)
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--iters", type=int, default=400)
    parser.add_argument("--burn", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument(
        "--out", type=Path, default=Path("surrogate_lowcount_summary.csv")
    )
    args = parser.parse_args()

    gh_x, gh_w = np.polynomial.hermite.hermgauss(31)
    rng = np.random.default_rng(args.seed)
    rows = []
    for scale in (2, 5):
        kappa = expected_kappa(scale, gh_x, gh_w)
        for n in args.counts:
            exact_all, surr_all, overlaps = [], [], []
            for _ in range(args.reps):
                w_true = rng.standard_normal(args.L)
                eps = rng.standard_normal((args.L, n))
                u = w_true[:, None] + eps
                th = THRESHOLDS[scale]
                ratings = np.searchsorted(th[1:-1], u) + 1
                exact = gibbs_posterior_means(
                    ratings, scale, args.iters, args.burn, rng
                )
                y = w_true[:, None] + eps / math.sqrt(kappa)
                post_prec = 1.0 + n * kappa
                surr = (kappa * y.sum(axis=1)) / post_prec
                exact_all.append(exact)
                surr_all.append(surr)
                top_e = set(np.argpartition(exact, -args.K)[-args.K:].tolist())
                top_s = set(np.argpartition(surr, -args.K)[-args.K:].tolist())
                overlaps.append(len(top_e & top_s) / args.K)
            e = np.concatenate(exact_all)
            s = np.concatenate(surr_all)
            rows.append(
                {
                    "scale": scale,
                    "responses_per_item": n,
                    "rmse": round(float(np.sqrt(((e - s) ** 2).mean())), 4),
                    "mae": round(float(np.abs(e - s).mean()), 4),
                    "corr": round(float(np.corrcoef(e, s)[0, 1]), 4),
                    "topk_overlap": round(float(np.mean(overlaps)), 3),
                    "reps": args.reps,
                }
            )
            print(rows[-1], flush=True)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        f.write(
            f"# L={args.L} K={args.K} reps={args.reps} gibbs_iters={args.iters} "
            f"burn={args.burn} seed={args.seed}; coupled shocks; N(0,1) prior; "
            "surrogate = information-equivalent Gaussian update.\n"
        )


if __name__ == "__main__":
    main()
