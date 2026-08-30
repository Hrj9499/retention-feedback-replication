# Reproducibility Manifest

This manifest maps the paper's numerical tables, figures, and compact
diagnostics to the script, seed, parameters, and output artifact used in the
current replication package.

## Environment

- Python: `Python 3.11.9`
- Install with pip:

```bash
python3 -m pip install -r requirements.txt
```

- Or create a conda environment:

```bash
conda env create -f environment.yml
conda activate optimal-retention-feedback
```

- Core packages:
  - `numpy==2.4.4`
  - `pandas==3.0.2`
  - `matplotlib==3.10.8`
  - `scipy==1.17.1`
- Environment snapshot: `ENVIRONMENT_LOCK.md` records the Python, core
  package, TeX, and smoke-test versions used for the packaged artifacts.
- TeX: `kpathsea version 6.4.2`; full source rebuild uses `latexmk`,
  `pdflatex`, and BibTeX. The package includes `retention.bbl` as a
  generated bibliography snapshot for environments where `pdflatex` is
  available but BibTeX is not.
- Working directory: run commands from the package root. The bundled
  `run_reproduction.sh` script changes into its own directory before
  executing the replication commands.

All numerical outputs can be regenerated with:

```bash
./run_reproduction.sh
```

The full script uses the published replication settings and may take tens of
minutes depending on hardware. For a quick smoke test of the pipeline, run:

```bash
./run_reproduction.sh --quick
```

Quick mode writes numerical outputs to `quick_outputs/`, uses `--worlds 2` for
the main synthetic benchmarks, the segment stress test, and the end-to-end EIS
workflow, and `--reps 2` for the MovieLens exercises, including the multi-split
robustness check. It is intended to verify that scripts run without overwriting
the published artifacts. It skips the PDF rebuild to avoid replacing
`retention.pdf`, and it does not reproduce the reported tables.

## Paper Build

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error retention.tex
```

Output:

- `retention.pdf`
- `retention.bbl`

## Synthetic Allocation Payoff

Paper location: Section 7.2, Table `Targeted allocation policies differ materially at matched budgets`.

```bash
python3 synthetic_allocation_benchmarks.py \
  --worlds 100 \
  --seed 20260406 \
  --output synthetic_allocation_summary.csv \
  --raw-output synthetic_allocation_raw.csv
```

Inputs and main settings:

- `L=60`, `K=6`, `S=5`
- Budgets: `10L`, `30L`, `100L`
- Warm start: five responses per item
- Priors: AR(1) with `rho=0.6`; independent covariance
- Policies: even, uncertainty-only, cutoff-only, combined cutoff x uncertainty,
  and the Proposition 4-inspired estimated next-answer-value benchmark.
  The next-answer-value benchmark is a derivative-style approximation
  used as a local diagnostic, not an exact evaluation of the directional
  derivative and not a globally optimal sequential policy.

Outputs:

- `synthetic_allocation_summary.csv`: table means, standard errors, win rates, paired improvements, best policy by mean, and an ex post within-family calibration reference. The `*_win_rate` columns report the fraction of worlds in which that policy ties for the smallest gap within the five-policy family at the same budget; paired improvements versus even allocation are reported separately.
- `synthetic_allocation_raw.csv`: one row per prior, world, budget, and policy.

## Cutoff Mechanism Figure

Paper location: Section 4.2, Figure `The cutoff mechanism`.

```bash
python3 mechanism_figure.py --seed 7 --out cutoff_mechanism.png
```

Deterministic illustration (one synthetic catalog); not an experiment.

## Policy Gap Comparison Figure

Paper location: Section 7.2, Figure of mean gaps by policy and budget.

```
python3 synthetic_allocation_benchmarks.py \
  --worlds 100 \
  --seed 20260406 \
  --budget-multiples "0,2,5,10,15,20,30,40,50,70,100" \
  --output synthetic_allocation_dense.csv \
  --raw-output synthetic_allocation_dense_raw.csv
python3 policy_gap_figure.py --input synthetic_allocation_dense.csv --output policy_gap_comparison.png
```

The dense run shares the world stream of the main benchmark, so its
10L/30L/100L checkpoints reproduce `synthetic_allocation_summary.csv`
exactly; the figure script is a deterministic rendering of the dense CSV.

## Refinement Boundary Map Figure

Paper location: Section 7.3, heatmap of the uncertainty-term gain.

```
python3 regime_map_figure.py --input combined_vs_cutoff_regime_map_summary.csv --output refinement_boundary_map.png
```

Deterministic aggregation of the published regime-map CSV; the three row
slices reproduce the slice means of the appendix regime-map table.

## Guard Ablation for the Cutoff Rule

Paper location: Section 7.2, ablation note on the guarded cutoff implementation.

```bash
python3 guard_ablation_benchmark.py \
  --worlds 100 \
  --seed 20260406 \
  --out guard_ablation_summary.csv
```

Same world-generation protocol, sizes, warm start, and budgets as the
main allocation benchmark. Policies: the published guarded cutoff-only
rule (proximity kernel with floor, divided by 1 + 0.10 x count), the
same rule without count damping, and the raw proximity kernel with
neither floor nor damping. Output:

- `guard_ablation_summary.csv`: mean gaps and standard errors per prior,
  budget, and ablation variant.

## Exact Theory Rule and SAA Reference

Paper location: Section 7.2 and Appendix "Greedy Versus Best Fixed Allocation".

```bash
python3 exact_rule_benchmark.py \
  --worlds 100 \
  --saa-worlds 30 \
  --seed 20260406 \
  --out exact_rule_summary.csv
```

Independent prior only (where the closed form applies); worlds match the
published independent-prior worlds of the main allocation benchmark.
Policies: the guarded cutoff-only rule, the exact closed-form one-step
rule (Proposition 4 evaluated exactly, no kernel), and a static SAA
reference optimized by multi-start SLSQP. Output:

- `exact_rule_summary.csv`: mean gaps and standard errors per policy and
  budget.

## OCBA-m Comparator

Paper location: Section 7.2, ranking-and-selection comparator note.

```bash
python3 ocba_benchmark.py --worlds 100 --seed 20260406 --out ocba_summary.csv
```

Sequential OCBA-m for optimal-subset selection (target shares from
current posterior means; delta floored at 5% of the cross-sectional
sd), against the guarded cutoff-only rule, on the main allocation
protocol. Output: `ocba_summary.csv`.

## Informative-Missingness Robustness

Paper location: Appendix "When Refinements Matter: Full Results" and the
conclusion's limitations paragraph.

```bash
python3 informative_missingness_benchmark.py --worlds 100 --seed 20260406   --out informative_missingness_summary.csv
```

Response probability 0.2 + 0.6*sigmoid(2(|y|-1)); prompts consume
budget whether or not an answer arrives; naive updating. Output:
`informative_missingness_summary.csv`.

## Low-Count Surrogate Audit

Paper location: Appendix "Surrogate-Fidelity Audit", low-count table.

```bash
python3 surrogate_lowcount_audit.py --reps 50 --seed 20260731   --out surrogate_lowcount_summary.csv
```

Self-contained coupled audit at one to five responses per item: exact
ordinal posterior means by data-augmentation Gibbs versus the
information-equivalent Gaussian update on the same latent shocks.
Output: `surrogate_lowcount_summary.csv`.

## Greedy Versus Best Fixed Allocation

Paper location: Section 6.3, quantitative note on the greedy rule.

```bash
python3 greedy_vs_optimal_benchmark.py \
  --worlds 20 \
  --seed 20260730 \
  --out greedy_vs_optimal_summary.csv
```

Inputs and main settings:

- `L=12`, `K=3`, budgets `2L` and `6L`, five-category information channel
- Priors: independent and AR(1) with `rho=0.6`
- Policies: even, exact one-step-lookahead greedy, combined cutoff x
  uncertainty rule, and a multi-start SLSQP search over fixed allocations
- Benchmark: the best fixed allocation found by any method on a
  common-random-numbers proxy objective

Outputs:

- `greedy_vs_optimal_summary.csv`: mean shortfall versus the best found
  allocation and the mean share of the achievable improvement over even
  allocation captured by each rule.

## Uncertainty-Refinement Stress Test

Paper location: Appendix "When Refinements Matter: Full Results", heterogeneous-information table.

```bash
python3 uncertainty_refinement_benchmark.py \
  --worlds 100 \
  --seed 20260528 \
  --output uncertainty_refinement_summary.csv \
  --raw-output uncertainty_refinement_raw.csv
```

Inputs and main settings:

- `L=60`, `K=6`
- Budgets of new responses: `L`, `3L`, `5L`
- Prior covariance: AR(1) with `rho=0.6`
- Warm start: deliberately heterogeneous information stocks near the cutoff.
  Some borderline products have many initial observations, while nearby
  rivals have only one.
- Policies: even, uncertainty-only, boundary-only, cutoff x uncertainty.
  Boundary-only uses cutoff proximity but deliberately drops the marginal
  uncertainty term so the diagnostic isolates why the refinement matters.

Outputs:

- `uncertainty_refinement_summary.csv`: mean gaps, standard errors, paired
  improvement of cutoff x uncertainty versus boundary-only, and pairwise win
  rates.
- `uncertainty_refinement_raw.csv`: one row per world, budget, and policy.

## Combined-vs-Cutoff Regime Map

Paper location: Appendix "When Refinements Matter: Full Results" (summarized in Section 7.3).

```bash
python3 combined_vs_cutoff_regime_map.py \
  --worlds 100 \
  --seed 20260529 \
  --output combined_vs_cutoff_regime_map_summary.csv \
  --raw-output combined_vs_cutoff_regime_map_raw.csv
```

Inputs and main settings:

- `L=60`, `K=6`
- Prior covariance: AR(1) with `rho=0.6`
- Budgets of new responses: `L`, `3L`, `10L`, `30L`
- Prior mean scenarios: separated, moderate, and crowded cutoff regions
- Initial information stocks: balanced warm start and heterogeneous near-cutoff
  stocks
- Information scales: low, baseline, and high per-answer information
- Policies: cutoff-only and cutoff x uncertainty

Outputs:

- `combined_vs_cutoff_regime_map_summary.csv`: mean gaps, standard errors,
  paired improvement of cutoff x uncertainty versus cutoff-only, pairwise win
  rates, percent improvement, and best policy by mean for each regime.
- `combined_vs_cutoff_regime_map_raw.csv`: one row per density scenario,
  information-stock scenario, information scale, world, budget, and policy.

## Archived Diagnostic Tables

Paper locations: supplementary numerical diagnostics referenced from Section 7.

```bash
python3 archived_diagnostic_tables.py
```

Output:

- `diagnostic_provenance.csv`: one row per compact diagnostic artifact, stating
  its manuscript location, evidentiary role, reproducibility level, and package
  boundary.
- `surrogate_fidelity_summary.csv`: RMSE, MAE, Pearson correlation, and top-K overlap values in the surrogate-fidelity table.
- `surrogate_topk_overlap_summary.csv`: top-K overlap subset retained for backward compatibility with earlier packages.
- `mean_sensitivity_summary.csv`: prior-mean sensitivity by distance-to-cutoff bin.
- `scale_comparative_summary.csv`: scale comparative-statics table in long format.
- `covariance_sensitivity_summary.csv`: covariance-sensitivity table.
- `factorization_diagnostic_summary.csv`: per-answer information factorization diagnostics.

This script materializes compact archived calibration summaries; it is not a
full stochastic replay of the earlier calibration runs. These diagnostics are
included so the manuscript tables have explicit CSV artifacts and can be
checked against the text. The headline allocation, refinement-map, crowding,
EIS, and MovieLens experiments have their own experiment scripts and raw
outputs below.

## Cutoff-Crowding Stress Test

Paper location: Appendix "When Refinements Matter: Full Results" (summarized in Section 7.3).

```bash
python3 crowded_cutoff_benchmarks.py \
  --worlds 100 \
  --seed 20260511 \
  --output crowded_cutoff_summary.csv \
  --raw-output crowded_cutoff_raw.csv
```

Inputs and main settings:

- `L=60`, `K=6`, `S=5`
- Budgets: `10L`, `30L`
- Warm start: five responses per item
- Prior covariance: AR(1) with `rho=0.6`
- Prior mean scenarios: separated cutoff, moderately crowded cutoff, crowded cutoff

Outputs:

- `crowded_cutoff_summary.csv`: means, standard errors, win rates, paired improvements, best policy by mean. The `*_win_rate` columns report the fraction of worlds in which that policy ties for the smallest gap within the four-policy family at the same budget; paired improvements versus even allocation are reported separately.
- `crowded_cutoff_raw.csv`: one row per scenario, world, budget, and policy.

## Correlation Spillover Diagnostic

Paper location: Appendix "When Refinements Matter: Full Results" (summarized in Section 7.3).

```bash
python3 correlation_spillover_benchmark.py \
  --worlds 200 \
  --seed 20260512 \
  --output correlation_spillover_summary.csv \
  --raw-output correlation_spillover_raw.csv
```

Inputs and main settings:

- `L=40`, `K=6`, `rho=0.7`
- AR(1) random-catalog diagnostic: budgets `10L`, `30L`; warm start of
  five responses per item.
- Clustered-substitute diagnostic: budgets `L`, `3L`, `6L`; block
  covariance with within-cluster correlation `0.88`; several near-cutoff
  substitutes share covariance with off-boundary proxy items that are
  relatively informative to query.
- Policies: even allocation, diagonal cutoff uncertainty, full-covariance spillover cutoff score

Outputs:

- `correlation_spillover_summary.csv`: mean gaps, standard errors, paired improvements, and full-spillover win rate versus the diagonal cutoff rule.
- `correlation_spillover_raw.csv`: one row per scenario, world, budget, and policy.

## Segment Allocation Stress Test

Paper location: Appendix "When Refinements Matter: Full Results" (summarized in Section 7.3).

```bash
python3 segment_allocation_stress_test.py \
  --worlds 100 \
  --seed 20260530 \
  --output segment_allocation_summary.csv \
  --raw-output segment_allocation_raw.csv
```

Inputs and main settings:

- Two segments with `L=40`, `K=5`, and total additional-response budget `80`.
- Segment A has decision weight `0.70`, a settled cutoff, prior standard
  deviation `0.35`, and 20 warm-start responses per item.
- Segment B has decision weight `0.30`, a crowded cutoff, prior standard
  deviation `0.95`, and 2 warm-start responses per item.
- Policies: weight-only, unweighted cutoff-only, segment-weighted cutoff, and
  segment-weighted local marginal.

Outputs:

- `segment_allocation_summary.csv`: weighted mean gaps, standard errors, share
  of budget sent to the smaller crowded segment, paired improvements versus
  weight-only allocation, and win rates.
- `segment_allocation_raw.csv`: one row per world and policy.

## EIS Format Validation

Paper location: Section 7.4, Table `EIS is a useful screening rule in these synthetic burden scenarios`.

```bash
python3 eis_format_validation.py \
  --worlds 200 \
  --seed 7 \
  --mc-draws 1200 \
  --output eis_format_validation_summary.csv \
  --raw-output eis_format_validation_raw.csv
```

Outputs:

- `eis_format_validation_summary.csv`
- `eis_format_validation_raw.csv`: one row per scenario, simulation world, and candidate format, including EIS score, realized proxy value, ranks, selected/best indicators, and regret. The value-unit selection shortfall quoted in Section 7.4 is the per-world best minus selected realized value from this file.

## Calibrated EIS Illustration

Paper location: Section 5.2, Table `Calibrated EIS profile`.

```bash
python3 eis_calibrated_illustration.py \
  --data-path data/ml-100k/u.data \
  --out eis_calibrated_summary.csv
```

Requires the MovieLens 100K `u.data` file (fetched by the MovieLens
benchmark step, or point `--data-path` at an existing copy). Outputs:

- `eis_calibrated_summary.csv`: EIS per format and burden-sensitivity
  setting, with fitted ordered-probit cutpoints, the empirical
  \(\beta(5)\), the item-level information-ratio spread, and the
  completion-anchor conventions recorded as trailing comments.

## End-to-End EIS Allocation Workflow

Paper location: Section 7.4, Table `End-to-end EIS plus allocation workflow`.

```bash
python3 eis_allocation_workflow.py \
  --worlds 100 \
  --seed 20260529 \
  --output eis_allocation_workflow_summary.csv \
  --raw-output eis_allocation_workflow_raw.csv \
  --format-output eis_allocation_workflow_format_raw.csv
```

Inputs and main settings:

- `L=60`, `K=10`, `T=30`
- Prior covariance: AR(1) with `rho=0.6`
- Common warm start: five baseline responses per item, using the 5-point
  information-equivalent channel
- Candidate formats and low/moderate/high burden completion profiles match
  the EIS validation exercise.
- Format rules: EIS, most raw answers, richest/longest format, highest
  completion format, and ex post oracle format.
- Allocation policies: uniform, cutoff-only, and cutoff x uncertainty.

Outputs:

- `eis_allocation_workflow_summary.csv`: scenario-level mean gaps, standard
  errors, selected-format modes, regret versus the ex post best format under
  the same allocation policy, and paired improvements against EIS+uniform and
  most-answers+cutoff-only baselines.
- `eis_allocation_workflow_raw.csv`: one row per burden scenario, synthetic
  world, and workflow rule.
- `eis_allocation_workflow_format_raw.csv`: one row per burden scenario,
  synthetic world, candidate format, and allocation policy.

## MovieLens Real-Rating Check

Paper location: Section 7.5, Table `MovieLens paired improvements over uniform sampling on one clean chronological split`.

```bash
python3 movielens_proxy_benchmarks.py \
  --top-l 200 \
  --k 10 \
  --history-share 0.20 \
  --feedback-share 0.50 \
  --reps 100 \
  --seed 20260327 \
  --summary-output movielens_experiment_summary.csv \
  --raw-output movielens_experiment_raw.csv \
  --split-output movielens_split_manifest.csv \
  --catalog-output movielens_selected_catalog.csv \
  --plot-output movielens_gap_vs_budget.png
```

Inputs and main settings:

- Dataset: MovieLens 100K from GroupLens. The raw MovieLens files are
  not redistributed in this replication package. If `data/ml-100k/u.data`
  is absent, the script downloads `ml-100k.zip` from the official
  GroupLens URL (`https://files.grouplens.org/datasets/movielens/ml-100k.zip`)
  before running. Users are responsible for complying with the GroupLens
  usage license, including the restriction on redistributing the raw data
  without separate permission.
- Checksum enforcement: the script verifies the downloaded `ml-100k.zip`
  checksum before extraction and verifies `u.data` before use. A mismatch
  stops the run. Downloads use standard certificate verification; if Python's
  downloader fails in a restricted certificate environment, the script falls
  back to `curl` with certificate verification still enabled.
- Verification-only mode: use `python3 movielens_proxy_benchmarks.py
  --verify-only` to verify the checksum, deterministic history-only catalog,
  and feedback/evaluation split without running policy simulations or writing
  outputs. Use `--skip-download` to fail fast in offline environments instead
  of fetching the official archive; pair it with `--data-path /path/to/u.data`
  when the data have already been obtained under the GroupLens terms.
- Expected checksums for the official data used to generate the packaged
  outputs: `ml-100k.zip` SHA256
  `50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229`;
  `u.data` SHA256
  `06416e597f82b7342361e41163890c81036900f418ad91315590814211dca490`.
- Chronological split: earliest 20% as history; catalog selection uses only this history period.
- Catalog selection: items are sorted by history-period rating count descending, with ties broken by item ID ascending. The selected item IDs and per-period rating counts are archived in `movielens_selected_catalog.csv`.
- Feedback/evaluation split: the first half of the post-history stream is the feedback pool, and the final half is the evaluation holdout.
- Policies: uniform, uncertainty-only, cutoff-only, cutoff x uncertainty.

Outputs:

- `movielens_experiment_summary.csv`
- `movielens_experiment_raw.csv`
- `movielens_split_manifest.csv`
- `movielens_selected_catalog.csv`
- `movielens_gap_vs_budget.png`

Magnitude anchors quoted in Section 7.5 (oracle box averages 4.39 stars
per item; catalog holdout average 3.63): derived from `u.data` sorted by
timestamp, evaluation rows 60000--99999 per `movielens_split_manifest.csv`,
holdout means over the 200 catalog items in
`movielens_selected_catalog.csv`; the oracle value is the sum of the ten
largest holdout means (43.93).

## MovieLens Robustness Snapshot

Paper location: replication-package robustness snapshot referenced from Section 7.5.

```bash
python3 movielens_robustness.py \
  --reps 20 \
  --seed 20260416 \
  --output movielens_robustness_summary.csv
```

For a quick data/split check without simulations:

```bash
python3 movielens_robustness.py --verify-only
```

Output:

- `movielens_robustness_summary.csv`

## MovieLens Multi-Split Robustness

Paper location: Section 7.5, Table `MovieLens multi-split robustness across clean chronological configurations`.

```bash
python3 movielens_multisplit_robustness.py \
  --reps 20 \
  --seed 20260529 \
  --aggregate-output movielens_multisplit_aggregate.csv \
  --split-output movielens_multisplit_summary.csv \
  --raw-output movielens_multisplit_raw.csv
```

Inputs and main settings:

- Dataset and checksum protocol match the MovieLens portability check.
- Six clean chronological configurations are used:
  `h15_f50`, `h20_f50`, `h25_f50`, `h30_f50`, `h20_f40`, and `h20_f60`.
  The first number is the history share used to select and initialize the
  catalog; the second is the share of post-history ratings used as the feedback
  pool, with the remaining post-history ratings reserved for evaluation.
- Each configuration uses `L=200`, `K=10`, history-only catalog selection,
  disjoint feedback-pool and evaluation-holdout periods, and 20 randomized
  repetitions.
- Budgets: `M=1000` and `M=2000`.
- Policies: uniform, uncertainty-only, cutoff-only, and cutoff x uncertainty.

Outputs:

- `movielens_multisplit_aggregate.csv`: budget-level aggregate means and
  split-level variation across the six clean split configurations.
- `movielens_multisplit_summary.csv`: one row per split configuration and
  budget, including policy means, paired improvements, split metadata, and
  whether cutoff-aware allocation beats uniform sampling.
- `movielens_multisplit_raw.csv`: one row per split configuration, repetition,
  policy, and budget.

## External Data Access

The MovieLens 100K raw data are intentionally not included in this package
because the GroupLens license permits research use but does not permit raw-data
redistribution without separate permission. The packaged MovieLens CSV/PNG
files are derived experiment outputs. To regenerate them, run the MovieLens
commands above; the scripts will download the official GroupLens archive if the
local `data/ml-100k/u.data` file is absent.

## Additional Notes

- Synthetic allocation and cutoff-crowding experiments now include raw
  world-level CSVs, not only aggregate tables.
- The paper reports rounded values; CSV files retain full precision.
- The MovieLens exercise is an open-data portability check for the top-K
  boundary mechanism. It is not a field retention test; subscription-retention
  validation would require platform data linking feedback, shipments, and churn.
