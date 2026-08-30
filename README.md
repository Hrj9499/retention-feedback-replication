# Designing Customer Feedback for Subscription Curation — Replication Package

Code and data artifacts for the paper *Designing Customer Feedback for
Subscription Curation* by Renjun Hu, Hyun-Soo Ahn, and Stefanus Jasin.
(SSRN link to be added.)

## Quick start

```bash
conda env create -f environment.yml
conda activate retention-feedback
bash run_reproduction.sh          # full run, tens of minutes
bash run_reproduction.sh --quick  # smoke test, writes to quick_outputs/
```

The full run regenerates every summary CSV, the raw per-world outputs,
the mechanism figure, and (if a TeX toolchain is available) the
manuscript PDF. `REPRODUCIBILITY.md` documents each step, its expected
outputs, and the seeds.

## Data

- **MovieLens 100K** is not redistributed. The MovieLens scripts download
  the official archive from GroupLens automatically. If the download
  fails in your environment (for example a TLS interception proxy),
  download `ml-100k.zip` from
  <https://files.grouplens.org/datasets/movielens/ml-100k.zip>, unzip it
  to `data/ml-100k/`, and pass `--data-path data/ml-100k/u.data`. The
  scripts verify the archive checksum.
- The **industrial retention log** discussed in the paper's retention
  appendix is proprietary and is not part of this package; the paper
  reports only aggregate, relative quantities.

## Layout

| Piece | Contents |
| --- | --- |
| `run_reproduction.sh` | Single entry script, 20 numbered steps |
| `*.py` | One script per experiment (main allocation benchmark, EIS validation and workflow, guard ablation, exact myopic one-answer and SAA reference, OCBA-m comparator, informative missingness, surrogate low-count audit, greedy vs best fixed allocation, MovieLens experiments and split robustness, refinement diagnostics, calibrated EIS illustration, mechanism figure) |
| `*_summary.csv` | Published summary outputs behind every table in the paper |
| `retention.tex`, `references.bib`, `retention.bbl` | Manuscript source |
| `REPRODUCIBILITY.md` | Step-by-step mapping from scripts to paper tables, seeds, and expected outputs |
| `ENVIRONMENT_LOCK.md`, `environment.yml` | Environment specification |

Raw per-world outputs are written by the entry script alongside the
summaries (the `--raw-output` arguments); they are regenerated rather
than shipped.

## License

MIT (see `LICENSE`).
