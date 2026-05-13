#!/bin/bash
#SBATCH --job-name=ja_fh_gpu
#SBATCH --output=logs/jet_assignment_%j.out
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=mps:20

set -euo pipefail

PROJECT_DIR="/home/export/sdurgut/scratch/ttHbb_SPANet"
OPTIONS_FILE="${PROJECT_DIR}/options_files/ttHbb_fully_hadronic/jet_assignment/options_train_signal_full.json"
LOG_DIR="${PROJECT_DIR}/spanet_output/jet_assignment"
VENV_ACTIVATE="${PROJECT_DIR}/SPANet/spanet_trial1/bin/activate"

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs" "${LOG_DIR}"
source "${VENV_ACTIVATE}"
echo "Activated venv: ${VENV_ACTIVATE}"
which python
python -V
export PYTHONPATH="${PROJECT_DIR}/SPANet:${PYTHONPATH:-}"
echo "PYTHONPATH=${PYTHONPATH}"

python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
PY

python -m spanet.train \
  -of "${OPTIONS_FILE}" \
  --log_dir "${LOG_DIR}" \
  --name "full_signal_jet_assignment" \
  --time_limit 00:06:00:00 \
  --gpus 1 \
  --verbose
