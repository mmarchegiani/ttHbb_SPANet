#!/usr/bin/env bash
set -euo pipefail

# Default image (override with: CONTAINER_IMAGE=/path/to/image.sif ./open_lpc_container.sh)
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmsml/cmsml:latest}"

if [[ ! -e "${CONTAINER_IMAGE}" ]]; then
  echo "ERROR: Container image not found: ${CONTAINER_IMAGE}" >&2
  exit 1
fi

# Pick runtime (apptainer preferred, singularity fallback)
if [[ -n "${CONTAINER_RUNTIME:-}" ]]; then
  RUNTIME="${CONTAINER_RUNTIME}"
elif command -v apptainer >/dev/null 2>&1; then
  RUNTIME="apptainer"
elif command -v singularity >/dev/null 2>&1; then
  RUNTIME="singularity"
else
  echo "ERROR: Neither apptainer nor singularity found in PATH." >&2
  exit 1
fi

bind_args=()

add_bind() {
  local path="$1"
  local required="${2:-0}"

  [[ -z "${path}" ]] && return 0

  if [[ -e "${path}" ]]; then
    bind_args+=("-B" "${path}")
  elif [[ "${required}" == "1" ]]; then
    echo "ERROR: Required mount path not found: ${path}" >&2
    exit 1
  fi
}

# Minimal required mounts on LPC
required_mounts=(
  "/tmp"
  "/cvmfs"
)

# LPC-focused optional mounts (plus a few portable ones)
optional_mounts=(
  "/afs"
  "/cvmfs/cms.cern.ch"
  "/cvmfs/grid.cern.ch"
  "/eos/uscms"
  "/uscms"
  "/uscms_data"
  "/uscmst1b_scratch"
  "/etc/grid-security"
  "${XDG_RUNTIME_DIR:-}"
  "${HOME:-}"
  "${HOME:-}/scratch"
  "/uscms_data/d3/${USER}"
)

for path in "${required_mounts[@]}"; do
  add_bind "${path}" 1
done

for path in "${optional_mounts[@]}"; do
  add_bind "${path}" 0
done

echo "Using runtime: ${RUNTIME}"
echo "Using image:   ${CONTAINER_IMAGE}"

exec "${RUNTIME}" shell "${bind_args[@]}" --nv "${CONTAINER_IMAGE}"
