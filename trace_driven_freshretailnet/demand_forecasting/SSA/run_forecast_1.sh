#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# demand_forecasting/SSA/run_forecast_1.sh
#
# Portable experiment runner (NO slurm, NO task-id, NO --run-all)
#
# Example:
#   bash run_forecast_1.sh --forecast-models "SeasonalNaive,SSA" --input-sources "raw,TimesNet"
#
# This script is designed to live inside: demand_forecasting/SSA/
# It will auto-cd to its own directory once for robustness.
# ==============================================================================

SCRIPT_NAME="$(basename "$0")"
SSA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SSA_DIR}/../.." && pwd)"

# Always operate from SSA_DIR so relative paths inside SSA are stable.
cd "${SSA_DIR}"

FORECAST_MODELS_CSV=""
INPUT_SOURCES_CSV=""

usage() {
  cat <<EOF
Usage:
  bash ${SCRIPT_NAME} \\
    --forecast-models "A,B,..." \\
    --input-sources "X,Y,..."

Options:
  --forecast-models   Comma-separated list (required)
  --input-sources     Comma-separated list (required)
  -h, --help          Show help

Notes:
  - "raw" triggers RAW workflow (generate predictions + evaluate).
  - Non-"raw" triggers RECOVERED workflow (predict + evaluate on recovered input).
EOF
}

# ---- Sanity checks: ensure script is in the expected place ----
if [[ ! -f "${SSA_DIR}/ssa_forecasting.py" || ! -f "${SSA_DIR}/ssa_seasonal_naive.py" ]]; then
  echo "[ERROR] Cannot find ssa_forecasting.py / ssa_seasonal_naive.py in: ${SSA_DIR}"
  echo "        Please place this script under demand_forecasting/SSA/ and run it there."
  exit 1
fi

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --forecast-models)
      FORECAST_MODELS_CSV="${2:-}"
      shift 2
      ;;
    --input-sources)
      INPUT_SOURCES_CSV="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${FORECAST_MODELS_CSV}" || -z "${INPUT_SOURCES_CSV}" ]]; then
  echo "[ERROR] --forecast-models and --input-sources are required"
  usage
  exit 1
fi

# ------------------------------------------------------------------------------
# Parse CSV → arrays (strip spaces to allow "raw, TimesNet")
FORECAST_MODELS_CSV="${FORECAST_MODELS_CSV// /}"
INPUT_SOURCES_CSV="${INPUT_SOURCES_CSV// /}"

IFS=',' read -r -a FORECAST_MODELS <<< "${FORECAST_MODELS_CSV}"
IFS=',' read -r -a INPUT_SOURCES <<< "${INPUT_SOURCES_CSV}"

# Same as your original script
ALL_RECOVERY_MODELS=("TimesNet")

echo "========================================================"
echo "Repo root       : ${REPO_ROOT}"
echo "SSA dir         : ${SSA_DIR}"
echo "Forecast models : ${FORECAST_MODELS[*]}"
echo "Input sources   : ${INPUT_SOURCES[*]}"
echo "Total runs      : $(( ${#FORECAST_MODELS[@]} * ${#INPUT_SOURCES[@]} ))"
echo "========================================================"

for FORECAST_MODEL in "${FORECAST_MODELS[@]}"; do
  for INPUT_SOURCE in "${INPUT_SOURCES[@]}"; do

    echo "--------------------------------------------------------"
    echo "Running:"
    echo "  Forecast Model = ${FORECAST_MODEL}"
    echo "  Input Source   = ${INPUT_SOURCE}"
    echo "--------------------------------------------------------"

    if [[ "${INPUT_SOURCE}" == "raw" ]]; then
      # ======================= CASE 1: RAW =======================
      echo "[INFO] RAW input workflow"

      RECOVERY_MODEL_NAME="${ALL_RECOVERY_MODELS[0]}"
      RECOVERED_EVAL_PATH="${REPO_ROOT}/latent_demand_recovery/exp/demand/demand_eval_${RECOVERY_MODEL_NAME}.parquet"

      echo "[STEP 1/2] Generating predictions..."
      if [[ "${FORECAST_MODEL}" == "SSA" ]]; then
        python ssa_forecasting.py \
          --recovered_eval_path "${RECOVERED_EVAL_PATH}" \
          --recovery_model_name "${RECOVERY_MODEL_NAME}"
      elif [[ "${FORECAST_MODEL}" == "SeasonalNaive" ]]; then
        python ssa_seasonal_naive.py \
          --recovered_eval_path "${RECOVERED_EVAL_PATH}" \
          --recovery_model_name "${RECOVERY_MODEL_NAME}"
      else
        echo "[ERROR] Unknown FORECAST_MODEL: ${FORECAST_MODEL}"
        exit 1
      fi

    else
      # =================== CASE 2: RECOVERED =====================
      echo "[INFO] Recovered input workflow"

      RECOVERY_MODEL_NAME="${INPUT_SOURCE}"
      RECOVERED_DEMAND_PATH="${REPO_ROOT}/latent_demand_recovery/exp/demand/demand_${RECOVERY_MODEL_NAME}.parquet"
      RECOVERED_EVAL_PATH="${REPO_ROOT}/latent_demand_recovery/exp/demand/demand_eval_${RECOVERY_MODEL_NAME}.parquet"

      if [[ "${FORECAST_MODEL}" == "SSA" ]]; then
        python ssa_forecasting.py \
          --demand \
          --demand_path "${RECOVERED_DEMAND_PATH}" \
          --recovered_eval_path "${RECOVERED_EVAL_PATH}" \
          --recovery_model_name "${RECOVERY_MODEL_NAME}"
      elif [[ "${FORECAST_MODEL}" == "SeasonalNaive" ]]; then
        python ssa_seasonal_naive.py \
          --demand \
          --demand_path "${RECOVERED_DEMAND_PATH}" \
          --recovered_eval_path "${RECOVERED_EVAL_PATH}" \
          --train_ground_truth_path "${RECOVERED_DEMAND_PATH}" \
          --recovery_model_name "${RECOVERY_MODEL_NAME}"
      else
        echo "[ERROR] Unknown FORECAST_MODEL: ${FORECAST_MODEL}"
        exit 1
      fi
    fi

    echo "[DONE] ${FORECAST_MODEL} + ${INPUT_SOURCE}"
    echo

  done
done

echo "========================================================"
echo "All experiments finished at $(date)"
echo "========================================================"
