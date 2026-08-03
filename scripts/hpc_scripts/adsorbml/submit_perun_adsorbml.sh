#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# submit_perun_adsorbml.sh — submit ONE AdsorbML GPU step on Perun (login node)
# ============================================================================
# Runs a single AdsorbML pipeline step inside the aarch64 NGC container built by
# scripts/hpc_scripts/adsorbml/setup_perun_uma_env.sh. Submit step 1 first (it writes the manifest),
# then step 2. Step 3 (ranking) is CPU-only bookkeeping — run it locally, not here.
#
#   STEP=1 ACCOUNT=<proj> SIF_PATH=/project/<proj>/containers/pytorch-ngc.dir INCLUDE="Mo2N_*" \
#     bash scripts/hpc_scripts/adsorbml/submit_perun_adsorbml.sh
#   # ...wait for step 1 to finish, then:
#   STEP=2 ACCOUNT=<proj> SIF_PATH=/project/<proj>/containers/pytorch-ngc.dir \
#     bash scripts/hpc_scripts/adsorbml/submit_perun_adsorbml.sh
#   # ...then locally:
#   python scripts/adsorbml/3-extract_rank.py
#
# Env-var knobs (defaults):
#   STEP                (required) 1 (relax) or 2 (screen H* sites)
#   ACCOUNT             (required) Perun project id (from `sprojects`)
#   SIF_PATH            (required) container image built by setup_perun_uma_env.sh
#   UMA_PYTHON          ${HOME}/envs/uma/bin/python    venv python inside the container
#   HF_HOME             /project/${ACCOUNT}/hf_cache   warmed weight cache
#   PARTITION           gpu_short                      gpu_short|gpu_medium|gpu_long
#   TIME_LIMIT          12:00:00
#   GRES                gpu:1                          gpu:4 = a full node (auto-parallel)
#   CPUS_PER_TASK       16
#   MEM_PER_GPU         64G
#   ADSORBML_DATA_ROOT  <repo>/data                    outputs root (shared by steps 1 & 2)
#   INCLUDE             (empty = all)                  glob filter, step 1 only
#   WORKERS             (empty = auto)                 workers; auto = one per visible GPU
#   LOG_DIR             <repo>/data/outputs/perun_logs
#   JOB_NAME            adsorbml-s${STEP}-perun
#   CONTAINER_BIN       (auto-detected)                full path/name of singularity|apptainer
#                                                      to bypass Lmod entirely on the compute node
#                                                      (e.g. /apps/singularity_gpu/bin/singularity)
# ----------------------------------------------------------------------------

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # scripts/hpc_scripts/adsorbml -> repo root

STEP="${STEP:-}"
ACCOUNT="${ACCOUNT:-}"
SIF_PATH="${SIF_PATH:-}"
UMA_PYTHON="${UMA_PYTHON:-${HOME}/envs/uma/bin/python}"
PARTITION="${PARTITION:-gpu_short}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
GRES="${GRES:-gpu:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM_PER_GPU="${MEM_PER_GPU:-64G}"
ADSORBML_DATA_ROOT="${ADSORBML_DATA_ROOT:-${WORKFLOW_ROOT}/data}"
INCLUDE="${INCLUDE:-}"
WORKERS="${WORKERS:-}"
LOG_DIR="${LOG_DIR:-${WORKFLOW_ROOT}/data/outputs/perun_logs}"
JOB_NAME="${JOB_NAME:-adsorbml-s${STEP}-perun}"
HF_HOME="${HF_HOME:-/project/${ACCOUNT}/hf_cache}"
HF_OFFLINE="${HF_OFFLINE:-1}"
CONTAINER_BIN="${CONTAINER_BIN:-}"

die() { echo "[submit-perun] ERROR: $*" >&2; exit 1; }

# --- Validation -------------------------------------------------------------
if [[ "${STEP}" != "1" && "${STEP}" != "2" ]]; then
  die "STEP must be 1 or 2. Example: STEP=1 ACCOUNT=proj SIF_PATH=... bash $0"
fi
[[ -n "${ACCOUNT}" ]]  || die "ACCOUNT is required (from \`sprojects\`)."
[[ -n "${SIF_PATH}" ]] || die "SIF_PATH is required (build it with scripts/hpc_scripts/adsorbml/setup_perun_uma_env.sh)."
[[ -e "${SIF_PATH}" ]] || die "SIF_PATH not found: ${SIF_PATH}"   # -e: a sandbox is a dir

# Partition sanity check (only if sinfo is available on the login node).
if command -v sinfo >/dev/null 2>&1; then
  if ! sinfo -h -o "%P" | sed 's/*//g' | awk '{print $1}' | grep -Fxq "${PARTITION}"; then
    echo "[submit-perun] Invalid partition: ${PARTITION}" >&2
    echo "[submit-perun] Available:" >&2
    sinfo -h -o "%P" | sed 's/*//g' | sort -u >&2
    exit 1
  fi
fi

# Step 2 depends on step 1's manifest (see scripts/adsorbml/_common.py MANIFEST_CSV).
if [[ "${STEP}" == "2" ]]; then
  MANIFEST="${ADSORBML_DATA_ROOT}/adsorbml_manifest.csv"
  [[ -f "${MANIFEST}" ]] || die "Step 2 needs the manifest from step 1, not found:
       ${MANIFEST}
       Run step 1 first (same ADSORBML_DATA_ROOT), let it finish, then submit step 2."
fi

mkdir -p "${LOG_DIR}"

echo "[submit-perun] Submitting AdsorbML step ${STEP}"
echo "  account=${ACCOUNT}  partition=${PARTITION}  gres=${GRES}  time=${TIME_LIMIT}"
echo "  sif=${SIF_PATH}"
echo "  data_root=${ADSORBML_DATA_ROOT}  hf_home=${HF_HOME}"
[[ "${STEP}" == "1" && -n "${INCLUDE}" ]] && echo "  include=${INCLUDE}"
[[ -n "${WORKERS}" ]] && echo "  workers=${WORKERS}"

sbatch \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --job-name="${JOB_NAME}" \
  --chdir="${WORKFLOW_ROOT}" \
  --nodes=1 \
  --gres="${GRES}" \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --mem-per-gpu="${MEM_PER_GPU}" \
  --time="${TIME_LIMIT}" \
  --output="${LOG_DIR}/%x_%j.out" \
  --error="${LOG_DIR}/%x_%j.err" \
  --export=ALL,WORKFLOW_ROOT="${WORKFLOW_ROOT}",STEP="${STEP}",SIF_PATH="${SIF_PATH}",UMA_PYTHON="${UMA_PYTHON}",HF_HOME="${HF_HOME}",HF_OFFLINE="${HF_OFFLINE}",ADSORBML_DATA_ROOT="${ADSORBML_DATA_ROOT}",INCLUDE="${INCLUDE}",WORKERS="${WORKERS}",CONTAINER_BIN="${CONTAINER_BIN}" \
  "${WORKFLOW_ROOT}/scripts/hpc_scripts/adsorbml/perun_adsorbml_worker.sh"
