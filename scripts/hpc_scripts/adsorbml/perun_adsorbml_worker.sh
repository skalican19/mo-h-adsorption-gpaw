#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# perun_adsorbml_worker.sh — sbatch job body for one AdsorbML GPU step
# ============================================================================
# Submitted by scripts/hpc_scripts/adsorbml/submit_perun_adsorbml.sh. Runs the chosen pipeline step
# with the native aarch64 fairchem venv (built by setup_perun_uma_env.sh), OFFLINE
# against the warmed HF weight cache. Not meant to be run by hand — the submitter
# exports everything it needs.
# ----------------------------------------------------------------------------

WORKFLOW_ROOT="${WORKFLOW_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"  # scripts/hpc_scripts/adsorbml -> repo root
STEP="${STEP:-}"
UMA_PYTHON="${UMA_PYTHON:-${HOME}/envs/uma/bin/python}"
HF_HOME="${HF_HOME:-}"
HF_OFFLINE="${HF_OFFLINE:-1}"
ADSORBML_DATA_ROOT="${ADSORBML_DATA_ROOT:-${WORKFLOW_ROOT}/data}"
INCLUDE="${INCLUDE:-}"
WORKERS="${WORKERS:-}"
PYTHON_MODULE="${PYTHON_MODULE:-}"

die() { echo "[perun-worker] ERROR: $*" >&2; exit 1; }

# --- Resolve the step script ------------------------------------------------
case "${STEP}" in
  1) STEP_SCRIPT="scripts/adsorbml/1-relax_uma_omat.py" ;;
  2) STEP_SCRIPT="scripts/adsorbml/2-run_adsorbml.py" ;;
  *) die "STEP must be 1 or 2 (got '${STEP}')." ;;
esac
[[ -x "${UMA_PYTHON}" ]] || die "UMA_PYTHON not found/executable: ${UMA_PYTHON}
     Build the env first: scripts/hpc_scripts/adsorbml/setup_perun_uma_env.sh"

# --- Load the venv's base Python module -------------------------------------
# A venv built from a site Python module needs that module loaded at runtime to
# find libpython. setup_perun_uma_env.sh recorded the module name next to the
# venv; use it (env override wins). `module` in a non-login sbatch subshell on
# aarch64 is often present-but-broken (wrong-arch Lmod tree), so re-source the
# init matching the LIVE launcher ($LMOD_CMD) before loading.
UMA_ENV_ROOT="$(cd "$(dirname "${UMA_PYTHON}")/.." && pwd)"
[[ -z "${PYTHON_MODULE}" && -r "${UMA_ENV_ROOT}/.python_module" ]] \
  && PYTHON_MODULE="$(<"${UMA_ENV_ROOT}/.python_module")"

module_available() { command -v module >/dev/null 2>&1 || type module >/dev/null 2>&1; }
ensure_module_fn() {
  local pkg="" init
  [[ -n "${LMOD_CMD:-}" ]] && pkg="$(dirname "$(dirname "${LMOD_CMD}")")"
  for init in \
      "${pkg:+${pkg}/init/bash}" \
      /apps/lmod_gpu/install/lmod/lmod/init/bash \
      "${LMOD_PKG:+${LMOD_PKG}/init/bash}" \
      "${MODULESHOME:+${MODULESHOME}/init/bash}" \
      /etc/profile.d/z00_lmod.sh \
      /etc/profile.d/lmod.sh \
      /etc/profile.d/modules.sh \
      /usr/share/lmod/lmod/init/bash; do
    [[ -n "${init}" && -r "${init}" ]] || continue
    set +u +e
    # shellcheck disable=SC1090
    source "${init}" >/dev/null 2>&1 || true
    set -u -e
    module_available && return 0
  done
  module_available
}

if [[ -n "${PYTHON_MODULE}" ]]; then
  if ensure_module_fn; then
    set +u +e
    module load "${PYTHON_MODULE}" >/dev/null 2>&1 || true
    set -u -e
  fi
fi

# --- Python args (step 2 has no --include; scope inherited via the manifest) -
py_args=()
if [[ "${STEP}" == "1" && -n "${INCLUDE}" ]]; then
  py_args+=(--include "${INCLUDE}")
fi
if [[ -n "${WORKERS}" ]]; then
  py_args+=(--workers "${WORKERS}")
fi

mkdir -p "${ADSORBML_DATA_ROOT}"
[[ -n "${HF_HOME}" ]] && mkdir -p "${HF_HOME}"

echo "[$(date --iso-8601=seconds)] AdsorbML step ${STEP} on $(hostname)"
echo "[perun-worker] python=${UMA_PYTHON}  module=${PYTHON_MODULE:-<none>}"
echo "[perun-worker] data_root=${ADSORBML_DATA_ROOT}  hf_home=${HF_HOME}  offline=${HF_OFFLINE}"
echo "[perun-worker] running ${STEP_SCRIPT} ${py_args[*]}"

export HF_HOME HF_HUB_OFFLINE="${HF_OFFLINE}" ADSORBML_DATA_ROOT
exec "${UMA_PYTHON}" -u "${WORKFLOW_ROOT}/${STEP_SCRIPT}" "${py_args[@]}"
