#!/usr/bin/env bash
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYENV_ROOT="${PYENV_ROOT:-${HOME}/.pyenv}"
ENV_NAME="${ENV_NAME:-cemea-env}"
MACHINE="${MACHINE:-devana}"
BOOTSTRAP_PREFIX="${BOOTSTRAP_PREFIX:-${HOME}/.local/mo_h_bootstrap}"
ACCOUNT="${ACCOUNT:-}"
PARTITION="${PARTITION:-cpu}" 
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-1}"
MEM_PER_CPU="${MEM_PER_CPU:-4G}"
JOB_NAME="${JOB_NAME:-gpaw-${MACHINE}}"
MANIFEST_DIR="${MANIFEST_DIR:-${WORKFLOW_ROOT}/data/outputs/slurm_manifests}"
MAX_HOURS_PER_STRUCTURE="${MAX_HOURS_PER_STRUCTURE:-0}"
RELAX_STEPS="${RELAX_STEPS:-8}"
FMAX="${FMAX:-0.10}"
KPTS="${KPTS:-}"

if [[ "${MACHINE}" != "devana" ]]; then
  echo "This script is DEVANA-only (SLURM array submission)." >&2
  echo "For desktop machines (node1/node2), run scripts/run_desktop_machine.sh instead." >&2
  exit 1
fi

if [[ -z "${ACCOUNT}" ]]; then
  echo "ACCOUNT is required. Example: ACCOUNT=myproject bash scripts/submit_devana_gpaw_array.sh" >&2
  exit 1
fi

if [[ -z "${PARTITION}" ]]; then
  echo "PARTITION is required (no default)." >&2
  if command -v sinfo >/dev/null 2>&1; then
    echo "Available partitions:" >&2
    sinfo -h -o "%P" | sed 's/*//g' | sort -u >&2
  fi
  exit 1
fi

if command -v sinfo >/dev/null 2>&1; then
  if ! sinfo -h -o "%P" | sed 's/*//g' | awk '{print $1}' | grep -Fxq "${PARTITION}"; then
    echo "Invalid partition: ${PARTITION}" >&2
    echo "Available partitions:" >&2
    sinfo -h -o "%P" | sed 's/*//g' | sort -u >&2
    exit 1
  fi
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

mkdir -p "${MANIFEST_DIR}"
MANIFEST_PATH="${MANIFEST_DIR}/${MACHINE}_$(date +%Y%m%d_%H%M%S).txt"

"${PYTHON_BIN}" "${WORKFLOW_ROOT}/scripts/gpaw_h_adsorption.py" \
  --machine "${MACHINE}" \
  --write-structure-list "${MANIFEST_PATH}"

TASK_COUNT="$(wc -l < "${MANIFEST_PATH}")"
if [[ "${TASK_COUNT}" -lt 1 ]]; then
  echo "Manifest is empty: ${MANIFEST_PATH}" >&2
  exit 1
fi

echo "Submitting ${TASK_COUNT} array tasks from ${MANIFEST_PATH}"

sbatch \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --job-name="${JOB_NAME}" \
  --chdir="${WORKFLOW_ROOT}" \
  --time="${TIME_LIMIT}" \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --mem-per-cpu="${MEM_PER_CPU}" \
  --array="1-${TASK_COUNT}" \
  --export=ALL,WORKFLOW_ROOT="${WORKFLOW_ROOT}",MANIFEST_PATH="${MANIFEST_PATH}",ENV_NAME="${ENV_NAME}",PYENV_ROOT="${PYENV_ROOT}",BOOTSTRAP_PREFIX="${BOOTSTRAP_PREFIX}",RELAX_STEPS="${RELAX_STEPS}",FMAX="${FMAX}",MAX_HOURS_PER_STRUCTURE="${MAX_HOURS_PER_STRUCTURE}",KPTS="${KPTS}" \
  "${WORKFLOW_ROOT}/scripts/devana_gpaw_array_worker.sh"