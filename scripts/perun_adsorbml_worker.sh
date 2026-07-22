#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# perun_adsorbml_worker.sh — sbatch job body for one AdsorbML GPU step
# ============================================================================
# Submitted by scripts/submit_perun_adsorbml.sh. Runs the chosen pipeline step
# inside the aarch64 NGC container, OFFLINE against the warmed HF weight cache.
# Not meant to be run by hand — the submitter exports everything it needs.
# ----------------------------------------------------------------------------

WORKFLOW_ROOT="${WORKFLOW_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
STEP="${STEP:-}"
SIF_PATH="${SIF_PATH:-}"
UMA_PYTHON="${UMA_PYTHON:-${HOME}/envs/uma/bin/python}"
HF_HOME="${HF_HOME:-}"
HF_OFFLINE="${HF_OFFLINE:-1}"
ADSORBML_DATA_ROOT="${ADSORBML_DATA_ROOT:-${WORKFLOW_ROOT}/data}"
INCLUDE="${INCLUDE:-}"
WORKERS="${WORKERS:-}"

die() { echo "[perun-worker] ERROR: $*" >&2; exit 1; }

# --- Resolve the step script ------------------------------------------------
case "${STEP}" in
  1) STEP_SCRIPT="scripts/adsorbml/1-relax_uma_omat.py" ;;
  2) STEP_SCRIPT="scripts/adsorbml/2-run_adsorbml.py" ;;
  *) die "STEP must be 1 or 2 (got '${STEP}')." ;;
esac
[[ -n "${SIF_PATH}" && -f "${SIF_PATH}" ]] || die "SIF_PATH not found: ${SIF_PATH}"

# --- Container runtime ------------------------------------------------------
if ! command -v singularity >/dev/null 2>&1 && ! command -v apptainer >/dev/null 2>&1; then
  if command -v module >/dev/null 2>&1 || type module >/dev/null 2>&1; then
    module load singularity 2>/dev/null || module load apptainer 2>/dev/null || true
  fi
fi
if   command -v singularity >/dev/null 2>&1; then CONTAINER_BIN=singularity
elif command -v apptainer   >/dev/null 2>&1; then CONTAINER_BIN=apptainer
else die "No singularity/apptainer found on the compute node."
fi

# --- Python args (step 2 has no --include; scope inherited via the manifest) -
py_args=()
if [[ "${STEP}" == "1" && -n "${INCLUDE}" ]]; then
  py_args+=(--include "${INCLUDE}")
fi
if [[ -n "${WORKERS}" ]]; then
  py_args+=(--workers "${WORKERS}")
fi

# --- Bind mounts (dedup; ensure each exists on the host) ---------------------
mkdir -p "${ADSORBML_DATA_ROOT}"
[[ -n "${HF_HOME}" ]] && mkdir -p "${HF_HOME}"
declare -A _seen=()
bind_args=()
for p in "${WORKFLOW_ROOT}" "${ADSORBML_DATA_ROOT}" "${HF_HOME}"; do
  [[ -z "${p}" ]] && continue
  if [[ -z "${_seen[$p]:-}" ]]; then
    bind_args+=(-B "${p}")
    _seen[$p]=1
  fi
done

echo "[$(date --iso-8601=seconds)] AdsorbML step ${STEP} on $(hostname)"
echo "[perun-worker] runtime=${CONTAINER_BIN}  sif=${SIF_PATH}"
echo "[perun-worker] data_root=${ADSORBML_DATA_ROOT}  hf_home=${HF_HOME}  offline=${HF_OFFLINE}"
echo "[perun-worker] running ${STEP_SCRIPT} ${py_args[*]}"

exec "${CONTAINER_BIN}" exec --nv \
  "${bind_args[@]}" \
  --env HF_HOME="${HF_HOME}",HF_HUB_OFFLINE="${HF_OFFLINE}",ADSORBML_DATA_ROOT="${ADSORBML_DATA_ROOT}" \
  "${SIF_PATH}" \
  "${UMA_PYTHON}" -u "${WORKFLOW_ROOT}/${STEP_SCRIPT}" "${py_args[@]}"
