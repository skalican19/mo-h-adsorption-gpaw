#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# perun_adsorbml_worker.sh — sbatch job body for one AdsorbML GPU step
# ============================================================================
# Submitted by scripts/hpc_scripts/adsorbml/submit_perun_adsorbml.sh. Runs the chosen pipeline step
# inside the aarch64 NGC container, OFFLINE against the warmed HF weight cache.
# Not meant to be run by hand — the submitter exports everything it needs.
# ----------------------------------------------------------------------------

WORKFLOW_ROOT="${WORKFLOW_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"  # scripts/hpc_scripts/adsorbml -> repo root
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
[[ -n "${SIF_PATH}" && -e "${SIF_PATH}" ]] || die "SIF_PATH not found: ${SIF_PATH}"   # -e: a sandbox is a dir

# --- Container runtime ------------------------------------------------------
# Resolve singularity/apptainer robustly in a non-login sbatch shell. On Perun's
# aarch64 GPU nodes the inherited `module` function is often present-but-BROKEN:
# it resolves against the wrong-arch Lmod tree (/apps/lmod [x86] vs
# /apps/lmod_gpu [aarch64]) and fails. So we NEVER trust the inherited function —
# we re-source the init matching the LIVE launcher ($LMOD_CMD) to install a
# known-good one, and fall back to the known binary paths if Lmod is unusable.
# Resolution order (mirrors setup_perun_uma_env.sh):
#   1. CONTAINER_BIN override (a name on PATH or a full path) — no module needed
#   2. already on PATH
#   3. re-source the arch-correct Lmod init, then load the versioned module
#   4. known Perun install paths (no Lmod needed; arch-agnostic)
# Override SINGULARITY_MODULE / APPTAINER_MODULE / CONTAINER_BIN if the site differs.
SINGULARITY_MODULE="${SINGULARITY_MODULE:-singularity/ce-4.4.1}"
APPTAINER_MODULE="${APPTAINER_MODULE:-apptainer}"
KNOWN_CONTAINER_BINS=(
  /apps/singularity_gpu/bin/singularity
  /apps/singularity/bin/singularity
  /apps/apptainer_gpu/bin/apptainer
  /apps/apptainer/bin/apptainer
)

module_available() { command -v module >/dev/null 2>&1 || type module >/dev/null 2>&1; }

ensure_module_fn() {
  # Derive the init tree from the live launcher first; it's the authoritative,
  # arch-correct match. $LMOD_PKG/$MODULESHOME can be stale (wrong-arch) here.
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
  # Nothing sourced cleanly — fall back to whatever was inherited.
  module_available
}

# 1. explicit override (full path, or a name already on PATH)
if [[ -n "${CONTAINER_BIN:-}" ]] && command -v "${CONTAINER_BIN}" >/dev/null 2>&1; then
  :  # trust it, skip the module machinery
# 2. already on PATH (e.g. a module the caller pre-loaded)
elif command -v singularity >/dev/null 2>&1; then CONTAINER_BIN=singularity
elif command -v apptainer   >/dev/null 2>&1; then CONTAINER_BIN=apptainer
else
  # 3. re-source the arch-correct Lmod init, then load the module
  if ensure_module_fn; then
    set +u +e
    module load "${SINGULARITY_MODULE}" >/dev/null 2>&1 \
      || module load singularity          >/dev/null 2>&1 \
      || module load "${APPTAINER_MODULE}" >/dev/null 2>&1 \
      || module load apptainer            >/dev/null 2>&1 || true
    set -u -e
  fi
  if   command -v singularity >/dev/null 2>&1; then CONTAINER_BIN=singularity
  elif command -v apptainer   >/dev/null 2>&1; then CONTAINER_BIN=apptainer
  else
    # 4. known Perun install locations (no Lmod needed; arch-agnostic)
    for _b in "${KNOWN_CONTAINER_BINS[@]}"; do
      if [[ -x "${_b}" ]]; then CONTAINER_BIN="${_b}"; break; fi
    done
  fi
  [[ -n "${CONTAINER_BIN:-}" ]] || die "No singularity/apptainer found on the compute node.
       Tried CONTAINER_BIN override, PATH, modules (${SINGULARITY_MODULE}, singularity, ${APPTAINER_MODULE}, apptainer), and ${KNOWN_CONTAINER_BINS[*]}.
       Set CONTAINER_BIN=/full/path/to/singularity to bypass modules entirely."
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
