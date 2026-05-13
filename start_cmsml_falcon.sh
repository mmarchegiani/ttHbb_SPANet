#!/usr/bin/env bash

CONTAINER_IMAGE="${CONTAINER_IMAGE:-/cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmsml/cmsml:latest}"

if [[ ! -e "${CONTAINER_IMAGE}" ]]; then
  echo "ERROR: Container image not found: ${CONTAINER_IMAGE}" >&2
  return 1 2>/dev/null || exit 1
fi

bind_args=()

# Required on this cluster.
required_mounts=(
  "/tmp"
  "/cvmfs"
)

# Optional mounts: keep CERN paths for portability, but only bind when present.
optional_mounts=(
  "/afs"
  "/cvmfs/cms.cern.ch"
  "/cvmfs/cms-griddata.cern.ch"
  "/cvmfs/cms-griddata.cern.ch/cat/metadata"
  "/eos/cms"
  "/eos/user/s/${USER}"
  "/eos/user/s/${USER}/ttHbb/ttHbb-fully-hadronic"
  "/etc/sysconfig/ngbauth-submit"
  "${XDG_RUNTIME_DIR:-}"
  "/home/export/${USER}"
  "/home/export/${USER}/scratch"
  "/home/export/${USER}/training_files"
)

for path in "${required_mounts[@]}"; do
  if [[ -e "${path}" ]]; then
    bind_args+=("-B" "${path}")
  else
    echo "ERROR: Required mount path not found: ${path}" >&2
    return 1 2>/dev/null || exit 1
  fi
done

for path in "${optional_mounts[@]}"; do
  if [[ -n "${path}" && -e "${path}" ]]; then
    bind_args+=("-B" "${path}")
  fi
done

apptainer shell "${bind_args[@]}" --nv "${CONTAINER_IMAGE}"

