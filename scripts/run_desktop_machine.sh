#!/usr/bin/env bash
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYENV_ROOT="${PYENV_ROOT:-${HOME}/.pyenv}"
ENV_NAME="${ENV_NAME:-cemea-env}"
MACHINE="${MACHINE:-node1}"
CORES_PER_CALC="${CORES_PER_CALC:-1}"
WORKERS="${WORKERS:-}"
RELAX_STEPS="${RELAX_STEPS:-8}"
FMAX="${FMAX:-0.10}"
MAX_HOURS_PER_STRUCTURE="${MAX_HOURS_PER_STRUCTURE:-0}"
KPTS="${KPTS:-}"

if [[ "${MACHINE}" != "node1" && "${MACHINE}" != "node2" && "${MACHINE}" != "node3" ]]; then
  echo "MACHINE must be node1, node2, or node3 for desktop execution." >&2
  echo "For DEVANA SLURM submission, use scripts/submit_devana_gpaw_array.sh (MACHINE=devana)." >&2
  exit 1
fi

if [[ -x "${PYENV_ROOT}/bin/pyenv" ]]; then
  export PATH="${PYENV_ROOT}/bin:${PATH}"
fi

if [[ -x "${PYENV_ROOT}/versions/${ENV_NAME}/bin/python" ]]; then
  PYTHON_BIN="${PYENV_ROOT}/versions/${ENV_NAME}/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

if [[ -z "${WORKERS}" ]]; then
  cpu_total="$(nproc 2>/dev/null || echo 1)"
  if [[ "${CORES_PER_CALC}" -le 0 ]]; then
    CORES_PER_CALC=1
  fi
  WORKERS="$(( cpu_total / CORES_PER_CALC ))"
  if [[ "${WORKERS}" -lt 1 ]]; then
    WORKERS=1
  fi
fi

echo "Running desktop machine split: ${MACHINE}"
echo "CPU total: $(nproc 2>/dev/null || echo unknown), CORES_PER_CALC=${CORES_PER_CALC}, workers=${WORKERS}"

cmd=(
  "${PYTHON_BIN}"
  "${WORKFLOW_ROOT}/scripts/gpaw_h_adsorption.py"
  --machine "${MACHINE}"
  --workers "${WORKERS}"
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
