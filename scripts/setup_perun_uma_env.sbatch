#!/usr/bin/env bash
#SBATCH --job-name=uma-setup
#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=uma-setup_%j.out
#SBATCH --error=uma-setup_%j.err

# ============================================================================
# setup_perun_uma_env.sbatch — run the one-time UMA env build as a BATCH job
# ============================================================================
# Fire-and-forget wrapper around scripts/setup_perun_uma_env.sh: builds the NGC
# container + fairchem venv and warms the gated uma-m-1p1 weight cache on a GPU
# node, so you don't have to babysit an interactive `srun` (which dies the
# moment you exit the shell or hit the walltime).
#
# Submit from the LOGIN node — needs your project id (--account) and HF token:
#   export HF_TOKEN=hf_xxxxxxxx
#   sbatch --account=<project_id> scripts/setup_perun_uma_env.sbatch
#
# Watch it:  squeue -u "$USER";  tail -f uma-setup_<jobid>.out
#
# Notes:
#   - GPU node = aarch64 + a GPU (the warm-up asserts torch.cuda.is_available()).
#   - Needs outbound internet on the node (image pull + pip + weight download).
#   - PROJECT_ID defaults to --account (SLURM_JOB_ACCOUNT); HF_TOKEN comes from
#     the environment (carried in by sbatch's default --export=ALL). The token is
#     NOT stored in this file — keep it out of version control.
#   - Idempotent: re-submit freely. It skips an existing image/venv and reuses
#     the layer + HF caches, so a timed-out run just resumes on the next submit.
#   - Override any setup knob by exporting it before sbatch, e.g.
#       SIF_PATH=... UMA_ENV=... TIME_LIMIT is a #SBATCH flag, pass on the CLI.
# ----------------------------------------------------------------------------
set -euo pipefail

# Run from the repo root (dir the script lives in, one level up from scripts/).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Project id: explicit PROJECT_ID wins, else the SLURM account this job runs under.
PROJECT_ID="${PROJECT_ID:-${SLURM_JOB_ACCOUNT:-}}"
if [[ -z "${PROJECT_ID}" ]]; then
  echo "[uma-setup] ERROR: no PROJECT_ID and no --account. Submit with:" >&2
  echo "            sbatch --account=<project_id> $0" >&2
  exit 1
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[uma-setup] WARNING: HF_TOKEN not set — the gated uma-m-1p1 warm-up will be" >&2
  echo "            SKIPPED and batch jobs will fail at model load. Re-submit with:" >&2
  echo "            export HF_TOKEN=hf_xxx ; sbatch --account=${PROJECT_ID} $0" >&2
fi

echo "[uma-setup] $(date -Is)  node=$(hostname)  arch=$(uname -m)  project=${PROJECT_ID}"

# The setup script reads HF_TOKEN from the environment on its own; don't pass it
# on the command line (keeps the token out of process listings / logs).
PROJECT_ID="${PROJECT_ID}" bash scripts/setup_perun_uma_env.sh
