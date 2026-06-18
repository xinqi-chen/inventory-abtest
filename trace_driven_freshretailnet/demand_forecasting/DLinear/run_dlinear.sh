#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# demand_forecasting/DLinear/run_dlinear.sh
#
# Portable runner for DLinear with input sources (raw / recovered)
#
# Example:
#   bash run_dlinear.sh --input-sources "raw,TimesNet"
#
# Assumption:
#   - Correct Python environment already activated.
# ==============================================================================

SCRIPT_NAME="$(basename "$0")"
D_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${D_DIR}/../.." && pwd)"

cd "${D_DIR}"

INPUT_SOURCES_CSV=""

usage() {
  cat <<EOF
Usage:
  bash ${SCRIPT_NAME} --input-sources "raw,TimesNet"

Options:
  --input-sources   Comma-separated list (required). e.g. "raw,TimesNet"
  -h, --help        Show help

Notes:
  - If input source == "raw":
      main.py trains/predicts on censored_dataset (once), then evaluates vs ALL_RECOVERY_MODELS.
  - Else (recovered input):
      main.py trains/predicts on recovered_dataset for that recovery model.
EOF
}

# ---- Sanity checks ----
[[ -f "${D_DIR}/main.py" ]] || { echo "[ERROR] Missing main.py in ${D_DIR}"; exit 1; }

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

ALL_RECOVERY_MODELS=("TimesNet" "ImputeFormer" "SAITS" "iTransformer" "CSDI")

mkdir -p results/logs
mkdir -p "${REPO_ROOT}/slurm_logs"  # 可删：保留你之前的目录习惯

echo "========================================================"
echo "Repo root      : ${REPO_ROOT}"
echo "DLinear dir    : ${D_DIR}"
echo "Input sources  : ${INPUT_SOURCES[*]}"
echo "========================================================"

run_raw() {
  echo "[STEP 1/2] Train + Predict on RAW (censored_dataset)"
  (python -u main.py \
    --is_training 1 \
    --train_only True \
    --model "DLinear" \
    --scale True \
    --loss 'mae' \
    --features "MS" \
    --data "censored_dataset" \
    --target "sale_amount" \
    --revin False \
    --total_seq_len 90 \
    --seq_len 62 \
    --pred_len 7 \
    --moving_avg 28 \
    --enc_in 7 \
    --des "censored_raw" \
    --do_predict \
    --itr 1 \
    --batch_size 1024 \
    --train_epochs 6 \
    --num_workers 0 \
    --learning_rate 0.001
  ) > results/logs/dlinear_prediction_generation_censored.log 2>&1

  echo "[STEP 2/2] Evaluate RAW predictions vs all recovered eval sets"
  for eval_model in "${ALL_RECOVERY_MODELS[@]}"; do
    local RECOVERED_EVAL_PATH="${REPO_ROOT}/latent_demand_recovery/exp/demand/demand_eval_${eval_model}.parquet"
    (python -u main.py \
      --is_training 0 \
      --model "DLinear" \
      --skip_predict \
      --scale True \
      --features "MS" \
      --data "censored_dataset" \
      --target "sale_amount" \
      --total_seq_len 90 \
      --seq_len 62 \
      --pred_len 7 \
      --moving_avg 28 \
      --enc_in 7 \
      --des "censored_raw" \
      --do_predict \
      --recovered_eval_path "${RECOVERED_EVAL_PATH}" \
      --itr 1
    ) > "results/logs/dlinear_metrics_evaluation_censored_vs_${eval_model}.log" 2>&1
  done
}

run_recovered() {
  local recovery_model="$1"
  local RECOVERED_DEMAND_PATH="${REPO_ROOT}/latent_demand_recovery/exp/demand/demand_${recovery_model}.parquet"
  local RECOVERED_EVAL_PATH="${REPO_ROOT}/latent_demand_recovery/exp/demand/demand_eval_${recovery_model}.parquet"

  echo "[STEP 1/1] Train + Predict on RECOVERED input: ${recovery_model}"
  (python -u main.py \
    --is_training 1 \
    --train_only True \
    --model "DLinear" \
    --scale True \
    --loss 'mae' \
    --features "MS" \
    --data "recovered_dataset" \
    --data_path "${RECOVERED_DEMAND_PATH}" \
    --target "sale_amount_pred" \
    --revin False \
    --total_seq_len 90 \
    --seq_len 62 \
    --pred_len 7 \
    --moving_avg 28 \
    --enc_in 7 \
    --des "recovered_${recovery_model}" \
    --do_predict \
    --recovered_eval_path "${RECOVERED_EVAL_PATH}" \
    --itr 1 \
    --batch_size 1024 \
    --train_epochs 6 \
    --num_workers 0 \
    --learning_rate 0.001
  ) > "results/logs/dlinear_recovered_${recovery_model}.log" 2>&1
}

for INPUT_SOURCE in "${INPUT_SOURCES[@]}"; do
  echo "--------------------------------------------------------"
  echo "Running DLinear + ${INPUT_SOURCE}"
  echo "--------------------------------------------------------"

  if [[ "${INPUT_SOURCE}" == "raw" ]]; then
    run_raw
  else
    run_recovered "${INPUT_SOURCE}"
  fi

  echo "[DONE] DLinear + ${INPUT_SOURCE}"
  echo
done

echo "========================================================"
echo "DLinear runs finished at $(date)"
echo "========================================================"
