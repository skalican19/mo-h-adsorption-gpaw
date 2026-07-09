#!/usr/bin/env bash
set -euo pipefail

WORKFLOW_ROOT="${WORKFLOW_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MANIFEST_PATH="${MANIFEST_PATH:-}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-}}"
ENV_NAME="${ENV_NAME:-cemea-env}"
PYENV_ROOT="${PYENV_ROOT:-${HOME}/.pyenv}"
BOOTSTRAP_PREFIX="${BOOTSTRAP_PREFIX:-${HOME}/.local/mo_h_bootstrap}"
CORES_PER_CALC="${SLURM_CPUS_PER_TASK:-${CORES_PER_CALC:-1}}"
RELAX_STEPS="${RELAX_STEPS:-8}"
FMAX="${FMAX:-0.10}"
MAX_HOURS_PER_STRUCTURE="${MAX_HOURS_PER_STRUCTURE:-0}"
KPTS="${KPTS:-}"

if [[ -z "${MANIFEST_PATH}" ]]; then
  echo "MANIFEST_PATH is required" >&2
  exit 1
fi

if [[ -z "${TASK_ID}" ]]; then
  echo "SLURM_ARRAY_TASK_ID or TASK_ID is required" >&2
  exit 1
fi

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "Manifest not found: ${MANIFEST_PATH}" >&2
  exit 1
fi

if [[ -x "${PYENV_ROOT}/bin/pyenv" ]]; then
  export PATH="${PYENV_ROOT}/bin:${PATH}"
fi

export LD_LIBRARY_PATH="${BOOTSTRAP_PREFIX}/lib64:${BOOTSTRAP_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

if [[ -x "${PYENV_ROOT}/versions/${ENV_NAME}/bin/python" ]]; then
  PYTHON_BIN="${PYENV_ROOT}/versions/${ENV_NAME}/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

STRUCTURE_NAME="$(sed -n "${TASK_ID}p" "${MANIFEST_PATH}")"
if [[ -z "${STRUCTURE_NAME}" ]]; then
  echo "No structure found for manifest line ${TASK_ID}" >&2
  exit 1
fi

echo "[$(date --iso-8601=seconds)] Task ${TASK_ID}: ${STRUCTURE_NAME}"
echo "[$(date --iso-8601=seconds)] Using ${CORES_PER_CALC} CPU(s) per task"

cmd=(
  "${PYTHON_BIN}"
  -u
  "${WORKFLOW_ROOT}/scripts/gpaw_h_adsorption.py"
  --structure-name "${STRUCTURE_NAME}"
  --workers 1
  --cores-per-calc "${CORES_PER_CALC}"
  --relax-steps "${RELAX_STEPS}"
  --fmax "${FMAX}"
)

if [[ "${MAX_HOURS_PER_STRUCTURE}" != "0" ]]; then
  cmd+=(--max-hours-per-structure "${MAX_HOURS_PER_STRUCTURE}")
fi

if [[ -n "${KPTS}" ]]; then
  cmd+=(--kpts "${KPTS}")
fi

exec "${cmd[@]}"