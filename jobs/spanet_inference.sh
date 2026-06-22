#!/usr/bin/env bash
# Condor worker wrapper for SPANet ONNX classification inference on one slice.
#
# Invoked by `apply_spanet_classification.py submit` with positional args:
#   $1 SCRIPT      absolute path to apply_spanet_classification.py
#   $2 H5          input HDF5 file
#   $3 CONFIG      SPANet event.yaml
#   $4 ONNX        ONNX model file
#   $5 START       slice start index
#   $6 STOP        slice stop index
#   $7 OUT_NPY     output .npy path for this slice
#   $8 BATCH_SIZE  inference batch size
#
# Environment (set by the submit file):
#   PROJECT_DIR    repo root (used for `pip install -e .` if no venv given)
#   VENV_ACTIVATE  optional path to a venv activate script to source
set -euo pipefail

date
hostname
echo "CONDOR_SCRATCH_DIR=${_CONDOR_SCRATCH_DIR:-<unset>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

SCRIPT="$1"; H5="$2"; CONFIG="$3"; ONNX="$4"
START="$5"; STOP="$6"; OUT_NPY="$7"; BATCH_SIZE="$8"

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${SCRIPT}")/../../.." && pwd)}"

# Activate a venv if provided, otherwise create a throwaway one with access to
# the system site-packages (cmsml image) and install the project.
if [[ -n "${VENV_ACTIVATE:-}" && -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck source=/dev/null
  source "${VENV_ACTIVATE}"
  echo "Activated venv: ${VENV_ACTIVATE}"
else
  echo "No venv provided; creating local venv with system site-packages."
  python -m venv myenv --system-site-packages
  # shellcheck source=/dev/null
  source myenv/bin/activate

  # Install onnxruntime-gpu version matched to the worker's CUDA runtime.
  # onnxruntime-gpu==1.16.x links against libcublasLt.so.11 (CUDA 11).
  # onnxruntime-gpu>=1.18   links against libcublasLt.so.12 (CUDA 12).
  # Pre-install before `pip install -e .` so the project's unversioned
  # onnxruntime-gpu requirement is satisfied by the right wheel.
  if ldconfig -p 2>/dev/null | grep -q 'libcublasLt.so.12'; then
    echo "CUDA 12 runtime detected -> installing onnxruntime-gpu>=1.18,<2"
    pip install 'onnxruntime-gpu>=1.18,<2'
  elif ldconfig -p 2>/dev/null | grep -q 'libcublasLt.so.11'; then
    echo "CUDA 11 runtime detected -> installing onnxruntime-gpu>=1.16,<1.17"
    pip install 'onnxruntime-gpu>=1.16,<1.17'
  else
    echo "No CUDA runtime detected -> installing CPU-only onnxruntime"
    pip install 'onnxruntime'
  fi

  pip install -e "${PROJECT_DIR}"
fi

which python || true
python -V

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "===== Allocated GPU(s) ====="
  nvidia-smi || true
  echo "============================"
else
  echo "WARNING: nvidia-smi not available; inference will fall back to CPU."
fi

echo "===== CUDA runtime libraries ====="
nvcc --version 2>/dev/null || echo "nvcc not in PATH"
ls /usr/local/cuda/version* 2>/dev/null || echo "No /usr/local/cuda/version file"
ldconfig -p 2>/dev/null | grep -E 'libcudart|libcublas|libcublasLt|libcudnn' | sort || echo "No CUDA runtime libs found via ldconfig"
echo "=================================="

echo "===== onnxruntime providers ====="
python -c "import onnxruntime as ort; print('ORT version:', ort.__version__); print('Build info:', ort.get_build_info()); print('Available providers:', ort.get_available_providers())"
echo "================================="

python "${SCRIPT}" infer \
  --h5 "${H5}" \
  --config "${CONFIG}" \
  --onnx "${ONNX}" \
  --start "${START}" \
  --stop "${STOP}" \
  --out-npy "${OUT_NPY}" \
  --batch-size "${BATCH_SIZE}"

echo "Inference slice [${START}, ${STOP}) -> ${OUT_NPY}"
date
