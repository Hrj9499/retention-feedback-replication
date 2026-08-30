#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODE="full"
if [[ "${1:-}" == "--quick" ]]; then
  MODE="quick"
  shift
fi
if [[ "$#" -ne 0 ]]; then
  echo "Usage: $0 [--quick]" >&2
  exit 2
fi

if [[ "$MODE" == "quick" ]]; then
  SYNTHETIC_WORLDS=2
  UNCERTAINTY_WORLDS=2
  COMBINED_WORLDS=2
  CROWDED_WORLDS=2
  CORRELATION_WORLDS=5
  SEGMENT_WORLDS=2
  EIS_WORKFLOW_WORLDS=2
  GREEDY_VS_OPT_WORLDS=2
  GUARD_ABLATION_WORLDS=2
  EXACT_SAA_WORLDS=2
  LOWCOUNT_REPS=3
  MOVIELENS_REPS=2
  ROBUSTNESS_REPS=2
  MULTISPLIT_REPS=2
  EIS_VALIDATION_WORLDS=3
  OUT_DIR="quick_outputs"
  mkdir -p "$OUT_DIR"
  echo "Running quick smoke-test reproduction. Outputs will be written to ${OUT_DIR}/ and will not overwrite the published artifacts."
else
  SYNTHETIC_WORLDS=100
  UNCERTAINTY_WORLDS=100
  COMBINED_WORLDS=100
  CROWDED_WORLDS=100
  CORRELATION_WORLDS=200
  SEGMENT_WORLDS=100
  EIS_WORKFLOW_WORLDS=100
  GREEDY_VS_OPT_WORLDS=20
  GUARD_ABLATION_WORLDS=100
  EXACT_SAA_WORLDS=30
  LOWCOUNT_REPS=50
  MOVIELENS_REPS=30
  ROBUSTNESS_REPS=20
  MULTISPLIT_REPS=20
  EIS_VALIDATION_WORLDS=200
  OUT_DIR="."
  echo "Running full reproduction. Expected runtime depends on hardware and can take tens of minutes."
fi

echo "[1/20] Archived diagnostic tables"
python3 archived_diagnostic_tables.py \
  --output-dir "$OUT_DIR"

echo "[2/20] Synthetic allocation benchmark"
python3 synthetic_allocation_benchmarks.py \
  --worlds "$SYNTHETIC_WORLDS" \
  --seed 20260406 \
  --output "$OUT_DIR/synthetic_allocation_summary.csv" \
  --raw-output "$OUT_DIR/synthetic_allocation_raw.csv"

echo "[3/20] Uncertainty-refinement benchmark"
python3 uncertainty_refinement_benchmark.py \
  --worlds "$UNCERTAINTY_WORLDS" \
  --seed 20260528 \
  --output "$OUT_DIR/uncertainty_refinement_summary.csv" \
  --raw-output "$OUT_DIR/uncertainty_refinement_raw.csv"

echo "[4/20] Combined-vs-cutoff regime map"
python3 combined_vs_cutoff_regime_map.py \
  --worlds "$COMBINED_WORLDS" \
  --seed 20260529 \
  --output "$OUT_DIR/combined_vs_cutoff_regime_map_summary.csv" \
  --raw-output "$OUT_DIR/combined_vs_cutoff_regime_map_raw.csv"

echo "[5/20] Cutoff-crowding benchmark"
python3 crowded_cutoff_benchmarks.py \
  --worlds "$CROWDED_WORLDS" \
  --seed 20260511 \
  --output "$OUT_DIR/crowded_cutoff_summary.csv" \
  --raw-output "$OUT_DIR/crowded_cutoff_raw.csv"

echo "[6/20] Correlation spillover diagnostic"
python3 correlation_spillover_benchmark.py \
  --worlds "$CORRELATION_WORLDS" \
  --seed 20260512 \
  --output "$OUT_DIR/correlation_spillover_summary.csv" \
  --raw-output "$OUT_DIR/correlation_spillover_raw.csv"

echo "[7/20] Segment allocation stress test"
python3 segment_allocation_stress_test.py \
  --worlds "$SEGMENT_WORLDS" \
  --seed 20260530 \
  --output "$OUT_DIR/segment_allocation_summary.csv" \
  --raw-output "$OUT_DIR/segment_allocation_raw.csv"

echo "[8/20] EIS format validation"
python3 eis_format_validation.py \
  --worlds "$EIS_VALIDATION_WORLDS" \
  --seed 7 \
  --mc-draws 1200 \
  --output "$OUT_DIR/eis_format_validation_summary.csv" \
  --raw-output "$OUT_DIR/eis_format_validation_raw.csv"

echo "[9/20] End-to-end EIS allocation workflow"
python3 eis_allocation_workflow.py \
  --worlds "$EIS_WORKFLOW_WORLDS" \
  --seed 20260529 \
  --output "$OUT_DIR/eis_allocation_workflow_summary.csv" \
  --raw-output "$OUT_DIR/eis_allocation_workflow_raw.csv" \
  --format-output "$OUT_DIR/eis_allocation_workflow_format_raw.csv"

echo "[10/20] Guard ablation for the cutoff rule"
python3 guard_ablation_benchmark.py \
  --worlds "$GUARD_ABLATION_WORLDS" \
  --seed 20260406 \
  --out "$OUT_DIR/guard_ablation_summary.csv"

echo "[11/20] Exact theory rule and SAA reference"
python3 exact_rule_benchmark.py \
  --worlds "$GUARD_ABLATION_WORLDS" \
  --saa-worlds "$EXACT_SAA_WORLDS" \
  --seed 20260406 \
  --out "$OUT_DIR/exact_rule_summary.csv"

echo "[12/20] OCBA-m comparator"
python3 ocba_benchmark.py \
  --worlds "$GUARD_ABLATION_WORLDS" \
  --seed 20260406 \
  --out "$OUT_DIR/ocba_summary.csv"

echo "[13/20] Informative-missingness robustness"
python3 informative_missingness_benchmark.py \
  --worlds "$GUARD_ABLATION_WORLDS" \
  --seed 20260406 \
  --out "$OUT_DIR/informative_missingness_summary.csv"

echo "[14/20] Low-count surrogate audit"
python3 surrogate_lowcount_audit.py \
  --reps "$LOWCOUNT_REPS" \
  --seed 20260731 \
  --out "$OUT_DIR/surrogate_lowcount_summary.csv"

echo "[15/20] Greedy versus best fixed allocation"
python3 greedy_vs_optimal_benchmark.py \
  --worlds "$GREEDY_VS_OPT_WORLDS" \
  --seed 20260730 \
  --out "$OUT_DIR/greedy_vs_optimal_summary.csv"

echo "[16/20] MovieLens real-rating check"
python3 movielens_proxy_benchmarks.py \
  --top-l 200 \
  --k 10 \
  --history-share 0.20 \
  --feedback-share 0.50 \
  --reps "$MOVIELENS_REPS" \
  --seed 20260327 \
  --summary-output "$OUT_DIR/movielens_experiment_summary.csv" \
  --raw-output "$OUT_DIR/movielens_experiment_raw.csv" \
  --split-output "$OUT_DIR/movielens_split_manifest.csv" \
  --catalog-output "$OUT_DIR/movielens_selected_catalog.csv" \
  --plot-output "$OUT_DIR/movielens_gap_vs_budget.png"

echo "[17/20] Calibrated EIS illustration"
python3 eis_calibrated_illustration.py \
  --data-path data/ml-100k/u.data \
  --out "$OUT_DIR/eis_calibrated_summary.csv"

echo "[18/20] MovieLens robustness snapshot"
python3 movielens_robustness.py \
  --reps "$ROBUSTNESS_REPS" \
  --seed 20260416 \
  --output "$OUT_DIR/movielens_robustness_summary.csv"

echo "[19/20] MovieLens multi-split robustness"
python3 movielens_multisplit_robustness.py \
  --reps "$MULTISPLIT_REPS" \
  --seed 20260529 \
  --aggregate-output "$OUT_DIR/movielens_multisplit_aggregate.csv" \
  --split-output "$OUT_DIR/movielens_multisplit_summary.csv" \
  --raw-output "$OUT_DIR/movielens_multisplit_raw.csv"

echo "[20/20] PDF build"
if [[ "$MODE" == "quick" ]]; then
  echo "Skipping PDF rebuild in quick mode to avoid overwriting retention.pdf. Use full mode or the Paper Build command for the manuscript PDF."
elif command -v latexmk >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error retention.tex
elif command -v pdflatex >/dev/null 2>&1 && [[ -f retention.bbl ]]; then
  echo "latexmk+bibtex not both available; using bundled retention.bbl with pdflatex fallback." >&2
  pdflatex -interaction=nonstopmode -halt-on-error retention.tex
  pdflatex -interaction=nonstopmode -halt-on-error retention.tex
  pdflatex -interaction=nonstopmode -halt-on-error retention.tex
else
  echo "TeX tools not found, or BibTeX is unavailable and retention.bbl is missing; numerical outputs were regenerated, but PDF compilation was skipped." >&2
fi
