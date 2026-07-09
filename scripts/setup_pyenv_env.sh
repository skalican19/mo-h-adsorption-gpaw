#!/usr/bin/env bash
set -euo pipefail

# Bootstrap this workflow on a fresh Linux node.
# Creates a pyenv-based environment and installs all required Python deps.
#
# Usage examples:
#   bash scripts/setup_pyenv_env.sh
#   ENV_NAME=cemea-env PYTHON_VERSION=3.10.16 bash scripts/setup_pyenv_env.sh
#   INSTALL_SYSTEM_DEPS=1 bash scripts/setup_pyenv_env.sh

ENV_NAME="${ENV_NAME:-cemea-env}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10.16}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQ_FILE="${PROJECT_ROOT}/requirements.txt"
INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-0}"
BOOTSTRAP_PREFIX="${BOOTSTRAP_PREFIX:-${HOME}/.local/mo_h_bootstrap}"
LIBFFI_VERSION="${LIBFFI_VERSION:-3.4.6}"
SQLITE_VERSION="${SQLITE_VERSION:-3460100}"
SQLITE_YEAR="${SQLITE_YEAR:-2024}"

log() {
  echo "[setup] $*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[setup] Missing command: $1" >&2
    exit 1
  fi
}

fetch_file() {
  local url="$1"
  local output="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$url" -o "$output"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$output" "$url"
  else
    echo "[setup] Need curl or wget to download $url" >&2
    exit 1
  fi
}

append_unique_flag() {
  local current="$1"
  local extra="$2"
  case " $current " in
    *" $extra "*) printf '%s' "$current" ;;
    *) printf '%s' "${current:+$current }$extra" ;;
  esac
}

build_rootless_deps() {
  local src_root="${BOOTSTRAP_PREFIX}/src"
  mkdir -p "${BOOTSTRAP_PREFIX}" "${src_root}"

  if [[ ! -f "${BOOTSTRAP_PREFIX}/lib/libffi.so" && ! -f "${BOOTSTRAP_PREFIX}/lib64/libffi.so" ]]; then
    local ffi_tar="${src_root}/libffi-${LIBFFI_VERSION}.tar.gz"
    local ffi_dir="${src_root}/libffi-${LIBFFI_VERSION}"
    log "Building local libffi ${LIBFFI_VERSION} under ${BOOTSTRAP_PREFIX}"
    fetch_file "https://github.com/libffi/libffi/releases/download/v${LIBFFI_VERSION}/libffi-${LIBFFI_VERSION}.tar.gz" "${ffi_tar}"
    rm -rf "${ffi_dir}"
    tar -xzf "${ffi_tar}" -C "${src_root}"
    (cd "${ffi_dir}" && ./configure --prefix="${BOOTSTRAP_PREFIX}" && make -j"$(command -v nproc >/dev/null 2>&1 && nproc || echo 4)" && make install)
  else
    log "Using existing local libffi from ${BOOTSTRAP_PREFIX}"
  fi

  if [[ ! -f "${BOOTSTRAP_PREFIX}/lib/libsqlite3.so" && ! -f "${BOOTSTRAP_PREFIX}/lib64/libsqlite3.so" ]]; then
    local sqlite_zip="${src_root}/sqlite-autoconf-${SQLITE_VERSION}.tar.gz"
    local sqlite_dir="${src_root}/sqlite-autoconf-${SQLITE_VERSION}"
    log "Building local sqlite ${SQLITE_VERSION} under ${BOOTSTRAP_PREFIX}"
    fetch_file "https://www.sqlite.org/${SQLITE_YEAR}/sqlite-autoconf-${SQLITE_VERSION}.tar.gz" "${sqlite_zip}"
    rm -rf "${sqlite_dir}"
    tar -xzf "${sqlite_zip}" -C "${src_root}"
    (cd "${sqlite_dir}" && ./configure --prefix="${BOOTSTRAP_PREFIX}" && make -j"$(command -v nproc >/dev/null 2>&1 && nproc || echo 4)" && make install)
  else
    log "Using existing local sqlite from ${BOOTSTRAP_PREFIX}"
  fi
}

configure_python_build_env() {
  local libdir="${BOOTSTRAP_PREFIX}/lib"
  local rpath_flags="-Wl,-rpath,${BOOTSTRAP_PREFIX}/lib"
  if [[ -d "${BOOTSTRAP_PREFIX}/lib64" ]]; then
    libdir="${BOOTSTRAP_PREFIX}/lib64:${BOOTSTRAP_PREFIX}/lib"
    rpath_flags="-Wl,-rpath,${BOOTSTRAP_PREFIX}/lib64 -Wl,-rpath,${BOOTSTRAP_PREFIX}/lib"
  fi

  export CPPFLAGS="$(append_unique_flag "${CPPFLAGS:-}" "-I${BOOTSTRAP_PREFIX}/include")"
  export LDFLAGS="$(append_unique_flag "${LDFLAGS:-}" "-L${BOOTSTRAP_PREFIX}/lib")"
  if [[ -d "${BOOTSTRAP_PREFIX}/lib64" ]]; then
    export LDFLAGS="$(append_unique_flag "${LDFLAGS:-}" "-L${BOOTSTRAP_PREFIX}/lib64")"
  fi
  for flag in ${rpath_flags}; do
    export LDFLAGS="$(append_unique_flag "${LDFLAGS:-}" "$flag")"
  done
  export PKG_CONFIG_PATH="${BOOTSTRAP_PREFIX}/lib/pkgconfig:${BOOTSTRAP_PREFIX}/lib64/pkgconfig:${PKG_CONFIG_PATH:-}"
  export LD_LIBRARY_PATH="${libdir}:${LD_LIBRARY_PATH:-}"
  export PYTHON_CONFIGURE_OPTS="${PYTHON_CONFIGURE_OPTS:-} --enable-shared"

  log "Configured Python build env with BOOTSTRAP_PREFIX=${BOOTSTRAP_PREFIX}"
}

python_has_required_stdlib() {
  local pybin="$1"
  "$pybin" - <<'PY' >/dev/null 2>&1
import ctypes
import sqlite3
print(ctypes, sqlite3)
PY
}

install_system_deps_debian() {
  log "Installing system packages (Debian/Ubuntu)..."
  sudo apt-get update
  sudo apt-get install -y \
    build-essential \
    gfortran \
    cmake \
    pkg-config \
    git \
    curl \
    wget \
    ca-certificates \
    libopenblas-dev \
    liblapack-dev \
    libfftw3-dev \
    libxc-dev \
    libscalapack-mpi-dev \
    libopenmpi-dev \
    openmpi-bin \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev
}

maybe_install_system_deps() {
  if [[ "${INSTALL_SYSTEM_DEPS}" != "1" ]]; then
    log "Skipping system package installation (set INSTALL_SYSTEM_DEPS=1 to enable)."
    return
  fi

  if ! command -v sudo >/dev/null 2>&1; then
    log "sudo not available; cannot auto-install system dependencies."
    return
  fi

  if command -v apt-get >/dev/null 2>&1; then
    install_system_deps_debian
  else
    log "Unsupported package manager for auto-install. Install compiler/BLAS/MPI deps manually."
  fi

  return
}

install_pyenv_if_missing() {
  export PYENV_ROOT="${PYENV_ROOT:-${HOME}/.pyenv}"

  if command -v pyenv >/dev/null 2>&1; then
    log "pyenv already installed: $(pyenv --version)"
    return
  fi

  if [[ -x "${PYENV_ROOT}/bin/pyenv" ]]; then
    export PATH="${PYENV_ROOT}/bin:${PATH}"
    eval "$(pyenv init -)"
    log "Using existing pyenv from ${PYENV_ROOT}"
    return
  fi

  log "pyenv not found. Installing pyenv into ~/.pyenv ..."
  require_cmd curl
  curl -fsSL https://pyenv.run | bash

  export PATH="${PYENV_ROOT}/bin:${PATH}"
  eval "$(pyenv init -)"

  log "pyenv installed."
}

init_pyenv_shell() {
  export PYENV_ROOT="${PYENV_ROOT:-${HOME}/.pyenv}"
  export PATH="${PYENV_ROOT}/bin:${PATH}"

  require_cmd pyenv
  eval "$(pyenv init -)"

  # Optional plugin init if available.
  if command -v pyenv-virtualenv-init >/dev/null 2>&1; then
    eval "$(pyenv virtualenv-init -)"
  fi
}

create_env() {
  build_rootless_deps
  configure_python_build_env

  log "Ensuring Python ${PYTHON_VERSION} exists in pyenv..."
  if [[ -x "${PYENV_ROOT}/versions/${PYTHON_VERSION}/bin/python" ]]; then
    if ! python_has_required_stdlib "${PYENV_ROOT}/versions/${PYTHON_VERSION}/bin/python"; then
      log "Existing pyenv Python ${PYTHON_VERSION} is missing ctypes/sqlite3; rebuilding it."
      rm -rf "${PYENV_ROOT}/versions/${PYTHON_VERSION}" "${PYENV_ROOT}/versions/${PYTHON_VERSION}/envs/${ENV_NAME}" "${PYENV_ROOT}/versions/${ENV_NAME}"
    fi
  fi

  pyenv install -s "${PYTHON_VERSION}"

  if ! python_has_required_stdlib "${PYENV_ROOT}/versions/${PYTHON_VERSION}/bin/python"; then
    echo "[setup] Python ${PYTHON_VERSION} built without ctypes/sqlite3 even after bootstrap deps." >&2
    echo "[setup] Check compiler output under ~/.pyenv/cache and ensure libffi/sqlite dev headers are visible." >&2
    exit 1
  fi

  if pyenv commands | grep -q '^virtualenv$'; then
    if ! pyenv versions --bare | grep -qx "${ENV_NAME}"; then
      log "Creating pyenv virtualenv: ${ENV_NAME}"
      pyenv virtualenv "${PYTHON_VERSION}" "${ENV_NAME}"
    else
      log "pyenv env ${ENV_NAME} already exists."
    fi
    PY_BIN="${PYENV_ROOT}/versions/${ENV_NAME}/bin/python"
    PIP_BIN="${PYENV_ROOT}/versions/${ENV_NAME}/bin/pip"
  else
    # Fallback: create a venv under pyenv versions path.
    if [[ ! -x "${PYENV_ROOT}/versions/${ENV_NAME}/bin/python" ]]; then
      log "pyenv-virtualenv plugin not found; creating venv at ${PYENV_ROOT}/versions/${ENV_NAME}"
      "${PYENV_ROOT}/versions/${PYTHON_VERSION}/bin/python" -m venv "${PYENV_ROOT}/versions/${ENV_NAME}"
    fi
    PY_BIN="${PYENV_ROOT}/versions/${ENV_NAME}/bin/python"
    PIP_BIN="${PYENV_ROOT}/versions/${ENV_NAME}/bin/pip"
  fi

  log "Using Python: ${PY_BIN}"
  "${PY_BIN}" --version
}

install_python_deps() {
  log "Upgrading pip/setuptools/wheel/cython..."
  "${PIP_BIN}" install --upgrade pip setuptools wheel cython

  log "Installing workflow requirements from ${REQ_FILE}"
  "${PIP_BIN}" install -r "${REQ_FILE}"
}

verify_install() {
  log "Verifying imports..."
  "${PY_BIN}" - <<'PY'
import importlib
mods = ["ctypes", "sqlite3", "ase", "numpy", "pandas", "gpaw", "pymatgen"]
for m in mods:
    importlib.import_module(m)
print("All required Python modules import successfully.")
PY

  log "Done. Activate with one of:"
  echo "  pyenv activate ${ENV_NAME}"
  echo "  source \"${PYENV_ROOT}/versions/${ENV_NAME}/bin/activate\""
  echo
  log "If your cluster strips library paths in batch/nohup shells, export this before running Python:"
  echo "  export LD_LIBRARY_PATH=\"${BOOTSTRAP_PREFIX}/lib64:${BOOTSTRAP_PREFIX}/lib:\${LD_LIBRARY_PATH:-}\""
}

main() {
  log "Project root: ${PROJECT_ROOT}"
  maybe_install_system_deps
  install_pyenv_if_missing
  init_pyenv_shell
  create_env
  install_python_deps
  verify_install
}

main "$@"
