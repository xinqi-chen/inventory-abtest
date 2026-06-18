#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PYTHON:-}" ]]; then
  PYTHON="python3"
fi
WORKERS="${WORKERS:-8}"
OUT_ROOT="${OUT_ROOT:-outputs_reproduced}"

cd "$(dirname "$0")"
mkdir -p "${OUT_ROOT}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$(pwd)/${OUT_ROOT}/.mplconfig}"
mkdir -p "${MPLCONFIGDIR}"

SCENARIO1_DIR="${OUT_ROOT}/scenario1_final"
SCENARIO2_DIR="${OUT_ROOT}/scenario2_final"
FINAL_DIR="${OUT_ROOT}/final_theory_verification"

"${PYTHON}" run_controlled_stochastic.py \
  --output-dir "${SCENARIO1_DIR}" \
  --workers "${WORKERS}" \
  --scenario 1 \
  --n 3000 \
  --t 60 \
  --r-gte 300 \
  --r-design 300 \
  --capacity-grid 0.90,0.92,1.20 \
  --delta-control -0.5 \
  --delta-treat -0.05 \
  --param-mode moderate_demand_heterogeneous \
  --assignment-mode bernoulli \
  --estimator diff_in_means \
  --skip-diagnostic-figures

"${PYTHON}" plot_abtest_violins.py \
  --output-dir "${SCENARIO1_DIR}" \
  --estimator diff_in_means

"${PYTHON}" run_controlled_stochastic.py \
  --output-dir "${SCENARIO2_DIR}" \
  --workers "${WORKERS}" \
  --scenario 2 \
  --n-list 3000 \
  --t-scenario2 60 \
  --r-gte-scenario2 300 \
  --r-design-scenario2 300 \
  --capacity-grid 0.85,1.00,1.10 \
  --capacity-grid-scenario2 0.85,1.00,1.10 \
  --eta-control 30.0 \
  --eta-treat 0.2 \
  --param-mode scenario2_clean_signal \
  --assignment-mode bernoulli \
  --estimator diff_in_means \
  --skip-diagnostic-figures

"${PYTHON}" plot_abtest_violins.py \
  --output-dir "${SCENARIO2_DIR}" \
  --estimator diff_in_means

"${PYTHON}" assemble_final_outputs.py \
  --scenario1-dir "${SCENARIO1_DIR}" \
  --scenario2-dir "${SCENARIO2_DIR}" \
  --output-dir "${FINAL_DIR}"

echo "Final outputs written to: ${FINAL_DIR}"
