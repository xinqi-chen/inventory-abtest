#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# demand_forecasting/TFT/run_tft.sh
#
# Portable runner for TFT with input sources (raw / recovered)
#
# Example:
#   bash run_tft.sh --input-sources "raw,TimesNet" --prediction-type deterministic
#
# Assumption:
#   - Correct Python environment already activated.
# ==============================================================================

SCRIPT_NAME="$(basename "$0")"
TFT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${TFT_DIR}/../.." && pwd)"

cd "${TFT_DIR}"

INPUT_SOURCES_CSV=""
PREDICTION_TYPE="deterministic"

usage() {
  cat <<EOF
Usage:
  bash ${SCRIPT_NAME} --input-sources "raw,TimesNet" [--prediction-type deterministic|quantile]
EOF
}

# ---- Sanity checks ----
[[ -f "${TFT_DIR}/trainTFT.py" ]] || { echo "[ERROR] Missing trainTFT.py"; exit 1; }
[[ -f "${TFT_DIR}/predictTFT.py" ]] || { echo "[ERROR] Missing predictTFT.py"; exit 1; }

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-sources)
      INPUT_SOURCES_CSV="${2:-}"
      shift 2
      ;;
    --prediction-type)
      PREDICTION_TYPE="${2:-}"
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

if [[ -z "${INPUT_SOURCES_CSV}" ]]; then
  echo "[ERROR] --input-sources is required"
  usage
  exit 1
fi

INPUT_SOURCES_CSV="${INPUT_SOURCES_CSV// /}"
IFS=',' read -r -a INPUT_SOURCES <<< "${INPUT_SOURCES_CSV}"

ALL_RECOVERY_MODELS=("TimesNet")

mkdir -p results/logs
mkdir -p "${REPO_ROOT}/slurm_logs"

echo "========================================================"
echo "Repo root     : ${REPO_ROOT}"
echo "TFT dir       : ${TFT_DIR}"
echo "Input sources : ${INPUT_SOURCES[*]}"
echo "Pred type     : ${PREDICTION_TYPE}"
echo "========================================================"

# run_raw() {
#   echo "[STEP 1/2] Train + Predict on RAW"
#   # (python trainTFT.py --recovery_model_name "raw" --prediction_type "${PREDICTION_TYPE}" &&
#   #  python predictTFT.py --recovery_model_name "raw" --prediction_type "${PREDICTION_TYPE}"
#   # ) > results/logs/TFT_prediction_generation_censored.log 2>&1
#   RECOVERED_TRAIN_PATH="${REPO_ROOT}/latent_demand_recovery/exp/demand/demand_${eval_model}.parquet"
#   RECOVERED_EVAL_PATH="${REPO_ROOT}/latent_demand_recovery/exp/demand/demand_eval_${eval_model}.parquet"

#   (python trainTFT.py --recovery_model_name "raw" --prediction_type "${PREDICTION_TYPE}" &&
#   python predictTFT.py --recovery_model_name "raw" \
#     --train_ground_truth_path "${RECOVERED_TRAIN_PATH}" \
#     --recovered_eval_path "${RECOVERED_EVAL_PATH}" \
#     --prediction_type "${PREDICTION_TYPE}"
#   ) > results/logs/TFT_prediction_generation_censored.log 2>&1

#   echo "[STEP 2/2] Evaluate RAW predictions vs all recovered eval sets"
#   for eval_model in "${ALL_RECOVERY_MODELS[@]}"; do
#     RECOVERED_EVAL_PATH="${REPO_ROOT}/latent_demand_recovery/exp/demand/demand_eval_${eval_model}.parquet"
#     (python predictTFT.py --skip_predict \
#       --recovered_eval_path "${RECOVERED_EVAL_PATH}" \
#       --recovery_model_name "${eval_model}" \
#       --prediction_type "${PREDICTION_TYPE}"
#     ) > "results/logs/TFT_metrics_evaluation_censored_vs_${eval_model}.log" 2>&1
#   done
# }

run_raw() {
  echo "[STEP 1/2] Train + Predict on RAW (predictions only)"
  (python trainTFT.py --recovery_model_name "raw" --prediction_type "${PREDICTION_TYPE}" &&
   python predictTFT.py --recovery_model_name "raw" --prediction_type "${PREDICTION_TYPE}"
  ) > results/logs/TFT_prediction_generation_censored.log 2>&1

  echo "[STEP 2/2] Evaluate RAW predictions vs all recovered (train+eval)"
  for eval_model in "${ALL_RECOVERY_MODELS[@]}"; do
    echo "[INFO] evaluating eval_model: ${eval_model}"
    RECOVERED_TRAIN_PATH="${REPO_ROOT}/latent_demand_recovery/exp/demand/demand_${eval_model}.parquet"
    RECOVERED_EVAL_PATH="${REPO_ROOT}/latent_demand_recovery/exp/demand/demand_eval_${eval_model}.parquet"

    (python predictTFT.py --skip_predict \
      --train_ground_truth_path "${RECOVERED_TRAIN_PATH}" \
      --recovered_eval_path "${RECOVERED_EVAL_PATH}" \
      --recovery_model_name "raw" \
      --prediction_type "${PREDICTION_TYPE}"
    ) > "results/logs/TFT_metrics_evaluation_censored_vs_${eval_model}.log" 2>&1
  done
}



run_recovered() {
  local RECOVERY_MODEL="$1"
  local RECOVERED_DEMAND_PATH="${REPO_ROOT}/latent_demand_recovery/exp/demand/demand_${RECOVERY_MODEL}.parquet"
  local RECOVERED_EVAL_PATH="${REPO_ROOT}/latent_demand_recovery/exp/demand/demand_eval_${RECOVERY_MODEL}.parquet"

  echo "[STEP 1/1] Train + Predict on RECOVERED input: ${RECOVERY_MODEL}"

  # === This block is intentionally kept to match the original slurm script ===
  (python trainTFT.py \
    --demand \
    --demand_path "${RECOVERED_DEMAND_PATH}" \
    --recovery_model_name "${RECOVERY_MODEL}" \
    --prediction_type "${PREDICTION_TYPE}" \
   && \
   python predictTFT.py \
    --demand \
    --demand_path "${RECOVERED_DEMAND_PATH}" \
    --recovered_eval_path "${RECOVERED_EVAL_PATH}" \
    --recovery_model_name "${RECOVERY_MODEL}" \
    --prediction_type "${PREDICTION_TYPE}" \
  ) > "results/logs/TFT_recovered_${RECOVERY_MODEL}.log" 2>&1
}

for INPUT_SOURCE in "${INPUT_SOURCES[@]}"; do
  echo "--------------------------------------------------------"
  echo "Running TFT + ${INPUT_SOURCE}"
  echo "--------------------------------------------------------"

  if [[ "${INPUT_SOURCE}" == "raw" ]]; then
    run_raw
  else
    run_recovered "${INPUT_SOURCE}"
  fi

  echo "[DONE] TFT + ${INPUT_SOURCE}"
  echo
done

echo "========================================================"
echo "TFT runs finished at $(date)"
echo "========================================================"
