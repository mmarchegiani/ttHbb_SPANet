#!/usr/bin/env bash
set -euo pipefail

date
hostname
uname -a
echo "CONDOR_SCRATCH_DIR=${_CONDOR_SCRATCH_DIR:-<unset>}"

# Defaults can be overridden from the submit file via environment.
PROJECT_DIR="${PROJECT_DIR:-/uscms/home/sdurgut1/nobackup/tthbb/ttHbb_SPANet}"
OPTIONS_FILE="${OPTIONS_FILE:-${PROJECT_DIR}/options_files/ttHbb_fully_hadronic/classifier/options_file_Run2_Run3_sig_QCD_classifier_btag_full.json}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/spanet_output/classifier_btag_full}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${PROJECT_DIR}/SPANet/myenv/bin/activate}"

if [[ ! -d "${PROJECT_DIR}" ]]; then
  echo "ERROR: PROJECT_DIR not found: ${PROJECT_DIR}" >&2
  echo "If this path is not visible on worker nodes, point PROJECT_DIR to a transferred/worker-visible location." >&2
  exit 2
fi

if [[ ! -f "${OPTIONS_FILE}" ]]; then
  echo "ERROR: OPTIONS_FILE not found: ${OPTIONS_FILE}" >&2
  exit 2
fi

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs" "${LOG_DIR}"

if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck source=/dev/null
  source "${VENV_ACTIVATE}"
  echo "Activated venv: ${VENV_ACTIVATE}"
else
  echo "WARNING: venv not found at ${VENV_ACTIVATE}; using current python environment."
fi

which python || true
python -V

export PYTHONPATH="${PROJECT_DIR}/SPANet:${PYTHONPATH:-}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "===== Allocated GPU(s) from nvidia-smi ====="
  nvidia-smi -L || true
  nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total,memory.free,compute_cap --format=csv,noheader || true
  echo "============================================="
else
  echo "WARNING: nvidia-smi not available in this runtime."
fi

python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    print("CUDA current device index:", dev)
    print("CUDA device name:", props.name)
    print("CUDA device total memory (GB):", round(props.total_memory / (1024 ** 3), 2))
    print("CUDA capability:", f"{props.major}.{props.minor}")
PY

python -m spanet.train \
  -of "${OPTIONS_FILE}" \
  --log_dir "${LOG_DIR}" \
  --name "classifier_btag_full" \
  --time_limit 07:00:00:00 \
  --verbose \
  --fp16 \
date
