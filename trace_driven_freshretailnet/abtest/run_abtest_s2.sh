#!/usr/bin/env bash
set -euo pipefail

# Robust (old-bash-friendly) shell-parallel runner for abtest/main_abtest_me.py (mode: s2)
# Usage:
#   bash abtest/run_abtest_s2.sh
# Overrides:
#   STOCKOUT_SUBSTITUTE=1 bash abtest/run_abtest_s2.sh
#   MAX_PROCS=4 N_JOBS=1 bash abtest/run_abtest_s2.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

BASE_EXPERIMENTS=(
  "TimesNet,TimesNet,Weekday,TimesNet,TFT"
  "TimesNet,TimesNet,Weekday,TimesNet,LightGBM"
  "TimesNet,TimesNet,Weekday,TimesNet,SSA"
)

CAPACITY_TYPES=("loose" "medium" "tight")

STOCKOUT_SUBSTITUTE="${STOCKOUT_SUBSTITUTE:-0}"
N_JOBS="${N_JOBS:-1}"
MAX_PROCS="${MAX_PROCS:-4}"
MODE="s2"

mkdir -p slurm_logs

mode_arg() {
  case "$1" in
    oracle) echo "--oracle_mode" ;;
    s1_oracle) echo "--s1_oracle_mode" ;;
    s1) echo "--s1_mode" ;;
    s2_oracle) echo "--s2_oracle_mode" ;;
    s2) echo "--s2_mode" ;;
    *) echo "" ;;
  esac
}
MODE_ARG="$(mode_arg "${MODE}")"

if [[ "${STOCKOUT_SUBSTITUTE}" != "0" && "${STOCKOUT_SUBSTITUTE}" != "1" ]]; then
  echo "[ERROR] STOCKOUT_SUBSTITUTE must be 0 or 1, got: ${STOCKOUT_SUBSTITUTE}"
  exit 1
fi
if ! [[ "${MAX_PROCS}" =~ ^[0-9]+$ ]] || (( MAX_PROCS < 1 )); then
  echo "[ERROR] MAX_PROCS must be a positive integer, got: ${MAX_PROCS}"
  exit 1
fi
if ! [[ "${N_JOBS}" =~ ^-?[0-9]+$ ]]; then
  echo "[ERROR] N_JOBS must be an integer, got: ${N_JOBS}"
  exit 1
fi

echo "========================================================"
echo "Running A/B tests (shell-parallel, no bash arrays)"
echo "Mode: ${MODE} | STOCKOUT_SUBSTITUTE=${STOCKOUT_SUBSTITUTE} | N_JOBS=${N_JOBS} | MAX_PROCS=${MAX_PROCS}"
echo "========================================================"

PIDS=""

prune_pids() {
  local kept=""
  local pid
  for pid in ${PIDS}; do
    if kill -0 "${pid}" 2>/dev/null; then
      kept="${kept} ${pid}"
    fi
  done
  PIDS="${kept}"
}

count_pids() {
  set -- ${PIDS}
  echo $#
}

wait_for_slot() {
  prune_pids
  while (( $(count_pids) >= MAX_PROCS )); do
    set -- ${PIDS}
    local pid0="$1"
    if [[ -z "${pid0}" ]]; then
      break
    fi
    if wait "${pid0}"; then
      :
    else
      echo "[ERROR] a background run failed (pid=${pid0}); check slurm_logs/"
      exit 1
    fi
    prune_pids
  done
}

for exp in "${BASE_EXPERIMENTS[@]}"; do
  IFS=',' read -r TRUE_DEMAND_SOURCE RECOVERY_CONTROL FORECAST_CONTROL RECOVERY_TREAT FORECAST_TREAT <<< "${exp}"

  for cap in "${CAPACITY_TYPES[@]}"; do
    LOG="slurm_logs/abtest_${MODE}_true-${TRUE_DEMAND_SOURCE}_cap-${cap}_ctrl-${RECOVERY_CONTROL}-${FORECAST_CONTROL}_treat-${RECOVERY_TREAT}-${FORECAST_TREAT}_sub-${STOCKOUT_SUBSTITUTE}.log"

    wait_for_slot

    (
      set -euo pipefail
      python -u abtest/main_abtest_me.py \
        --recovery_control "${RECOVERY_CONTROL}" \
        --forecast_control "${FORECAST_CONTROL}" \
        --recovery_treat "${RECOVERY_TREAT}" \
        --forecast_treat "${FORECAST_TREAT}" \
        --stockout_substitute "${STOCKOUT_SUBSTITUTE}" \
        --capacity_type "${cap}" \
        --n_jobs "${N_JOBS}" \
        --true_demand_source "${TRUE_DEMAND_SOURCE}" \
        ${MODE_ARG} \
        > "${LOG}" 2>&1
      echo "[DONE] wrote log: ${LOG}"
    ) &

    PIDS="${PIDS} $!"
  done
done

prune_pids
for pid in ${PIDS}; do
  wait "${pid}" || { echo "[ERROR] a run failed (pid=${pid}); check slurm_logs/"; exit 1; }
done

echo "========================================================"
echo "All s2 A/B tests finished at $(date)"
echo "========================================================"
