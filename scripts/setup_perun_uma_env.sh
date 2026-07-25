#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# setup_perun_uma_env.sh — build the fairchem/UMA GPU environment on Perun ONCE
# ============================================================================
# Perun GPU nodes are aarch64 (Grace-Hopper GH200); a plain `pip install torch`
# there is CPU-only and silently ignores the GPU. This script builds a
# CUDA-enabled aarch64 environment from an NVIDIA NGC PyTorch container, layers
# fairchem on top, and warms the (gated) UMA model weight cache so batch jobs
# can run OFFLINE against it.
#
# RUN THIS INTERACTIVELY ON A GPU NODE (it needs the aarch64 arch + a GPU + net):
#   srun --partition=gpu_short --gres=gpu:1 --time=02:00:00 --pty bash
#   PROJECT_ID=<proj> bash scripts/setup_perun_uma_env.sh --hf-token hf_xxx
#
# The UMA model (uma-m-1p1) is a GATED HuggingFace model. Before running:
#   1. Accept its license once on the HuggingFace model page.
#   2. Create a HF access token and pass it via --hf-token (or the HF_TOKEN env var).
#
# Env-var knobs (defaults):
#   PROJECT_ID   (required) Perun project id (from `sprojects`); used for HF_HOME
#   SIF_PATH     /project/${PROJECT_ID}/containers/pytorch-ngc.dir  SANDBOX image DIR (not a .sif)
#   NGC_TAG      25.01-py3                             nvcr.io/nvidia/pytorch tag (VERIFY a real one)
#   UMA_ENV      ${HOME}/envs/uma                      venv (reuses container torch)
#   HF_HOME      /project/${PROJECT_ID}/hf_cache       where weights are stored (NOT /home)
#   UMA_MODEL    uma-m-1p1                             model checkpoint to warm
#   SINGULARITY_MODULE  singularity/ce-4.4.1           Lmod module for the container runtime
#   APPTAINER_MODULE    apptainer                      Lmod module if the site ships apptainer
#   CONTAINER_BIN       (auto-detected)                full path/name of singularity|apptainer to bypass module logic
#   SINGULARITY_TMPDIR  /scratch/${PROJECT_ID}/sing_tmp    scratch for the image-pull unpack (default /tmp overflows)
#   SINGULARITY_CACHEDIR /scratch/${PROJECT_ID}/sing_cache downloaded-layer cache for the pull
# ----------------------------------------------------------------------------

HF_TOKEN_ARG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf-token) HF_TOKEN_ARG="${2:-}"; shift 2 ;;
    --hf-token=*) HF_TOKEN_ARG="${1#*=}"; shift ;;
    -h|--help)
      sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

PROJECT_ID="${PROJECT_ID:-}"
SIF_PATH="${SIF_PATH:-}"   # default set after PROJECT_ID is known (see below)
NGC_TAG="${NGC_TAG:-25.01-py3}"
UMA_ENV="${UMA_ENV:-${HOME}/envs/uma}"
UMA_MODEL="${UMA_MODEL:-uma-m-1p1}"
# Lmod won't resolve a bare `singularity` on Perun — it needs the versioned name.
SINGULARITY_MODULE="${SINGULARITY_MODULE:-singularity/ce-4.4.1}"
APPTAINER_MODULE="${APPTAINER_MODULE:-apptainer}"
# HF token: CLI arg wins, else fall back to the environment.
HF_TOKEN="${HF_TOKEN_ARG:-${HF_TOKEN:-}}"

log() { echo "[setup-perun] $*"; }
die() { echo "[setup-perun] ERROR: $*" >&2; exit 1; }

if [[ -z "${PROJECT_ID}" ]]; then
  die "PROJECT_ID is required (from \`sprojects\`). Example: PROJECT_ID=myproj bash $0 --hf-token hf_xxx"
fi
HF_HOME="${HF_HOME:-/project/${PROJECT_ID}/hf_cache}"
# Container image lives on /project (big, persistent), NOT /home (small NFS quota).
# It's a SANDBOX directory, not a .sif: Perun GPU-node kernels can't mount the
# .sif squashfs ("bad superblock … compression"); a plain dir needs no mount.
SIF_PATH="${SIF_PATH:-/project/${PROJECT_ID}/containers/pytorch-ngc.dir}"
# The NGC image is many GB; singularity unpacks it under $SINGULARITY_TMPDIR
# (default /tmp, RAM-backed on compute nodes → overflows mid-unpack). Redirect
# temp + layer cache to /scratch (Lustre: big, purged, not backed up — fine for
# throwaway build data).
SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-/scratch/${PROJECT_ID}/sing_tmp}"
SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-/scratch/${PROJECT_ID}/sing_cache}"

# --- 1. Guard: must be on an aarch64 GPU node -------------------------------
ARCH="$(uname -m)"
if [[ "${ARCH}" != "aarch64" ]]; then
  die "This is a ${ARCH} host. The UMA env must be built on an aarch64 GPU node.
       Grab one first:  srun --partition=gpu_short --gres=gpu:1 --time=02:00:00 --pty bash"
fi

# --- 2. Locate the container runtime ----------------------------------------
# `module` is an Lmod shell function. In a non-login `bash <script>` subshell it
# may be MISSING, or present-but-broken: on Perun's aarch64 GPU nodes an inherited
# `module` can point at the wrong-arch Lmod tree (/apps/lmod [x86] vs
# /apps/lmod_gpu [aarch64]) and fail. So we NEVER trust the inherited function —
# we re-source the init that matches the LIVE launcher ($LMOD_CMD) to install a
# known-good one. Lmod's init + generated eval trip `set -u`, so we relax
# nounset/errexit while sourcing and loading.
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

load_container_runtime() {
  # 1. explicit override (full path, or a name already on PATH)
  if [[ -n "${CONTAINER_BIN:-}" ]] && command -v "${CONTAINER_BIN}" >/dev/null 2>&1; then
    return
  fi
  # 2. already on PATH (e.g. a module the caller pre-loaded)
  if command -v singularity >/dev/null 2>&1; then CONTAINER_BIN=singularity; return; fi
  if command -v apptainer   >/dev/null 2>&1; then CONTAINER_BIN=apptainer;   return; fi
  # 3. Lmod (preferred over a raw binary: the module also sets up cache/bind env)
  if ensure_module_fn; then
    set +u +e
    module load "${SINGULARITY_MODULE}" >/dev/null 2>&1 \
      || module load singularity          >/dev/null 2>&1 \
      || module load "${APPTAINER_MODULE}" >/dev/null 2>&1 \
      || module load apptainer            >/dev/null 2>&1 || true
    set -u -e
  fi
  if command -v singularity >/dev/null 2>&1; then CONTAINER_BIN=singularity; return; fi
  if command -v apptainer   >/dev/null 2>&1; then CONTAINER_BIN=apptainer;   return; fi
  # 4. fallback: known Perun install locations (no Lmod needed; arch-agnostic)
  local b
  for b in "${KNOWN_CONTAINER_BINS[@]}"; do
    if [[ -x "${b}" ]]; then CONTAINER_BIN="${b}"; return; fi
  done
  die "No singularity/apptainer found.
       Tried: CONTAINER_BIN override, PATH, modules (${SINGULARITY_MODULE}, singularity, ${APPTAINER_MODULE}, apptainer), and ${KNOWN_CONTAINER_BINS[*]}.
       Set CONTAINER_BIN=/full/path/to/singularity to bypass, or SINGULARITY_MODULE=<name>."
}
load_container_runtime
log "Container runtime: ${CONTAINER_BIN}"

# --- 3. Build the NGC PyTorch image as a SANDBOX (dir) if missing -----------
# A --sandbox directory, NOT a .sif: Perun GPU-node kernels can't mount the .sif
# squashfs ("bad superblock … compression"), whereas a plain directory rootfs
# needs no mount. `singularity exec` treats a dir exactly like a .sif.
mkdir -p "$(dirname "${SIF_PATH}")"
if [[ -e "${SIF_PATH}" ]]; then
  log "Image already present: ${SIF_PATH} (skipping build)"
else
  # Give the multi-GB unpack room off /tmp (see config block above).
  mkdir -p "${SINGULARITY_TMPDIR}" "${SINGULARITY_CACHEDIR}"
  export SINGULARITY_TMPDIR SINGULARITY_CACHEDIR
  export APPTAINER_TMPDIR="${SINGULARITY_TMPDIR}" APPTAINER_CACHEDIR="${SINGULARITY_CACHEDIR}"
  log "Building nvcr.io/nvidia/pytorch:${NGC_TAG} -> ${SIF_PATH} (sandbox; needs internet on this node)"
  log "  tmpdir=${SINGULARITY_TMPDIR}  cachedir=${SINGULARITY_CACHEDIR}"
  "${CONTAINER_BIN}" build --sandbox "${SIF_PATH}" "docker://nvcr.io/nvidia/pytorch:${NGC_TAG}"
fi

# --- 4. Create a venv that reuses the container's CUDA torch ----------------
if [[ -x "${UMA_ENV}/bin/python" ]]; then
  log "venv already exists: ${UMA_ENV} (skipping create)"
else
  log "Creating venv (system-site-packages -> reuse container torch): ${UMA_ENV}"
  mkdir -p "$(dirname "${UMA_ENV}")"
  "${CONTAINER_BIN}" exec --nv "${SIF_PATH}" python -m venv --system-site-packages "${UMA_ENV}"
fi

# --- 5. Install fairchem into the venv (matches requirements.txt) -----------
log "Installing fairchem-core + fairchem-data-oc into the venv (needs internet)"
"${CONTAINER_BIN}" exec --nv "${SIF_PATH}" "${UMA_ENV}/bin/pip" install --upgrade pip
"${CONTAINER_BIN}" exec --nv "${SIF_PATH}" "${UMA_ENV}/bin/pip" install fairchem-core fairchem-data-oc

# --- 6. Verify GPU + warm the gated HF weight cache -------------------------
mkdir -p "${HF_HOME}"
if [[ -z "${HF_TOKEN}" ]]; then
  log "WARNING: no HF token provided (--hf-token / HF_TOKEN). Skipping weight warm-up."
  log "         The gated model '${UMA_MODEL}' will NOT be cached, and batch jobs will fail"
  log "         at model load. Accept the license on the HF model page, create a token, then re-run:"
  log "             PROJECT_ID=${PROJECT_ID} bash $0 --hf-token hf_xxx"
else
  log "Verifying GPU + downloading '${UMA_MODEL}' into ${HF_HOME} ..."
  "${CONTAINER_BIN}" exec --nv \
    -B "${HF_HOME}" \
    --env HF_HOME="${HF_HOME}",HF_TOKEN="${HF_TOKEN}",UMA_MODEL="${UMA_MODEL}" \
    "${SIF_PATH}" "${UMA_ENV}/bin/python" - <<'PY'
import os
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False — the env is CPU-only!"
print(f"[warmup] torch {torch.__version__}  CUDA {torch.version.cuda}  GPU {torch.cuda.get_device_name(0)}")
from fairchem.core import FAIRChemCalculator
model = os.environ["UMA_MODEL"]
FAIRChemCalculator.from_model_checkpoint(model, task_name="omat", device="cuda")
print(f"[warmup] '{model}' loaded and cached under {os.environ['HF_HOME']}")
PY
  log "Warm-up OK — weights cached."
fi

# --- 7. Summary: values to hand to the submitter ----------------------------
cat <<EOF

[setup-perun] Done. Pass these to scripts/submit_perun_adsorbml.sh:

    SIF_PATH=${SIF_PATH}
    UMA_PYTHON=${UMA_ENV}/bin/python
    HF_HOME=${HF_HOME}

Example (from the login node):
    STEP=1 ACCOUNT=${PROJECT_ID} SIF_PATH=${SIF_PATH} \\
      bash scripts/submit_perun_adsorbml.sh
EOF
