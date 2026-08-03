#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# setup_perun_uma_env.sh — build the fairchem/UMA GPU environment on Perun ONCE
# ============================================================================
# Perun GPU nodes are aarch64 (Grace-Hopper GH200). A plain `pip install torch`
# there is CPU-only and silently ignores the GPU; the CUDA-enabled aarch64
# torch 2.8 wheel fairchem needs lives ONLY on the cu129 index. This script
# builds a native venv from a site Python module, installs that exact wheel +
# fairchem, and warms the (gated) UMA weight cache so batch jobs run OFFLINE.
# (No container — see the "GPU env history" note in the /perun-hpc skill.)
#
# RUN THIS INTERACTIVELY ON A GPU NODE (it needs the aarch64 arch + a GPU + net):
#   srun --partition=gpu_short --gres=gpu:1 --time=02:00:00 --pty bash
#   PROJECT_ID=<proj> bash scripts/hpc_scripts/adsorbml/setup_perun_uma_env.sh --hf-token hf_xxx
#
# The UMA model (uma-m-1p1) is a GATED HuggingFace model. Before running:
#   1. Accept its license once on the HuggingFace model page.
#   2. Create a HF access token and pass it via --hf-token (or the HF_TOKEN env var).
#
# Env-var knobs (defaults):
#   PROJECT_ID     (required) Perun project id (from `sprojects`); used for HF_HOME
#   UMA_ENV        ${HOME}/envs/uma                    native venv location
#   HF_HOME        /project/${PROJECT_ID}/hf_cache     where weights are stored (NOT /home)
#   UMA_MODEL      uma-m-1p1                            model checkpoint to warm
#   PYTHON_MODULE  Python/3.12.3-GCCcore-13.3.0        site Python module (fairchem needs 3.11–3.13)
#   CUDA_MODULE    (empty)                             optional; only if a dep must compile CUDA from source
#   TORCH_SPEC     torch==2.8.0+cu129                  the ONLY aarch64 CUDA torch-2.8 wheel
#   TORCH_INDEX_URL https://download.pytorch.org/whl/cu129   index that carries it (cu128 skips 2.8)
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
UMA_ENV="${UMA_ENV:-${HOME}/envs/uma}"
UMA_MODEL="${UMA_MODEL:-uma-m-1p1}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}"
CUDA_MODULE="${CUDA_MODULE:-}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.8.0+cu129}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu129}"
# HF token: CLI arg wins, else fall back to the environment.
HF_TOKEN="${HF_TOKEN_ARG:-${HF_TOKEN:-}}"

log() { echo "[setup-perun] $*"; }
die() { echo "[setup-perun] ERROR: $*" >&2; exit 1; }

if [[ -z "${PROJECT_ID}" ]]; then
  die "PROJECT_ID is required (from \`sprojects\`). Example: PROJECT_ID=myproj bash $0 --hf-token hf_xxx"
fi
HF_HOME="${HF_HOME:-/project/${PROJECT_ID}/hf_cache}"

# --- 1. Guard: must be on an aarch64 GPU node -------------------------------
ARCH="$(uname -m)"
if [[ "${ARCH}" != "aarch64" ]]; then
  die "This is a ${ARCH} host. The UMA env must be built on an aarch64 GPU node.
       Grab one first:  srun --partition=gpu_short --gres=gpu:1 --time=02:00:00 --pty bash"
fi

# --- 2. Make `module` usable in this subshell, then load Python -------------
# `module` is an Lmod shell function. In a non-login `bash <script>` subshell it
# may be MISSING, or present-but-broken: on Perun's aarch64 GPU nodes an inherited
# `module` can point at the wrong-arch Lmod tree (/apps/lmod [x86] vs
# /apps/lmod_gpu [aarch64]) and fail. So we NEVER trust the inherited function —
# we re-source the init that matches the LIVE launcher ($LMOD_CMD) to install a
# known-good one. Lmod's init + generated eval trip `set -u`, so we relax
# nounset/errexit while sourcing and loading.
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

ensure_module_fn || die "Could not get a working \`module\` command on this node.
     Load your Python module by hand, then re-run with PYTHON_MODULE set to it."

set +u +e
module load "${PYTHON_MODULE}" 2>/dev/null
_mod_rc=$?
[[ -n "${CUDA_MODULE}" ]] && module load "${CUDA_MODULE}" 2>/dev/null
set -u -e
if [[ ${_mod_rc} -ne 0 ]] || ! command -v python >/dev/null 2>&1; then
  echo "[setup-perun] Could not load Python module '${PYTHON_MODULE}'." >&2
  echo "[setup-perun] Pick a real one (3.11–3.13) from:" >&2
  set +u +e; module avail Python 2>&1 | sed 's/^/    /' >&2; set -u -e
  die "Re-run with PYTHON_MODULE=<name> set to one of the above."
fi
log "Python: $(python -V 2>&1) via module ${PYTHON_MODULE}"

# --- 3. Create the native venv (isolated: no --system-site-packages) --------
if [[ -x "${UMA_ENV}/bin/python" ]]; then
  log "venv already exists: ${UMA_ENV} (skipping create)"
else
  log "Creating venv: ${UMA_ENV}"
  mkdir -p "$(dirname "${UMA_ENV}")"
  python -m venv "${UMA_ENV}"
fi
# shellcheck disable=SC1091
source "${UMA_ENV}/bin/activate"

# --- 4. Install the aarch64 CUDA torch, then fairchem -----------------------
# torch MUST go in first from the cu129 index — it's the only aarch64 build that
# satisfies fairchem's torch~=2.8.0 with CUDA. Installing fairchem first would
# pull a CPU-only torch off PyPI.
log "Installing ${TORCH_SPEC} from ${TORCH_INDEX_URL}"
python -m pip install --upgrade pip
python -m pip install --index-url "${TORCH_INDEX_URL}" "${TORCH_SPEC}"
log "Installing fairchem-core + fairchem-data-oc"
python -m pip install fairchem-core fairchem-data-oc

# --- 5. Verify GPU + warm the gated HF weight cache -------------------------
mkdir -p "${HF_HOME}"
if [[ -z "${HF_TOKEN}" ]]; then
  log "WARNING: no HF token provided (--hf-token / HF_TOKEN). Skipping weight warm-up."
  log "         The gated model '${UMA_MODEL}' will NOT be cached, and batch jobs will fail"
  log "         at model load. Accept the license on the HF model page, create a token, then re-run:"
  log "             PROJECT_ID=${PROJECT_ID} bash $0 --hf-token hf_xxx"
else
  log "Verifying GPU + downloading '${UMA_MODEL}' into ${HF_HOME} ..."
  HF_HOME="${HF_HOME}" HF_TOKEN="${HF_TOKEN}" UMA_MODEL="${UMA_MODEL}" \
    python - <<'PY'
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

# --- 6. Persist the Python module so the worker reloads the exact same one ---
# A venv built from a module Python needs that module loaded at runtime to find
# libpython; the sbatch worker reads this to `module load` the matching version.
printf '%s\n' "${PYTHON_MODULE}" > "${UMA_ENV}/.python_module"

# --- 7. Summary: values to hand to the submitter ----------------------------
cat <<EOF

[setup-perun] Done. Pass these to scripts/hpc_scripts/adsorbml/submit_perun_adsorbml.sh:

    UMA_PYTHON=${UMA_ENV}/bin/python
    HF_HOME=${HF_HOME}

Example (from the login node):
    STEP=1 ACCOUNT=${PROJECT_ID} UMA_PYTHON=${UMA_ENV}/bin/python \\
      bash scripts/hpc_scripts/adsorbml/submit_perun_adsorbml.sh
EOF
