#!/bin/bash
#SBATCH --job-name=merge_h5_cpu
#SBATCH --output=logs/merge_h5_cpu_rogue/result_merge_%j.out
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G

set -euo pipefail

SECONDS=0
JOB_START_EPOCH="$(date +%s)"
JOB_START_HUMAN="$(date '+%Y-%m-%d %H:%M:%S %Z')"

PROJECT_DIR="/home/export/sdurgut/scratch/ttHbb_SPANet"
CLASSIFICATION_DIR="/home/export/sdurgut/scratch/parquet_to_h5_output/classification/with_MASK"
MERGED_DIR="$CLASSIFICATION_DIR/merged_with_MASK"
mkdir -p "$MERGED_DIR"
VENV_ACTIVATE="$PROJECT_DIR/SPANet/spanet_env_cpu/bin/activate"
MERGE_SCRIPT="$PROJECT_DIR/scripts/dataset/merge_h5_files.py"
LOG_DIR="$PROJECT_DIR/logs/merge_h5_cpu_rogue/"
mkdir -p "$LOG_DIR"

SCRATCH_DIR="/mnt/scratch/$USER/$SLURM_JOB_ID"
mkdir -p "$SCRATCH_DIR"
echo "Job running on $(hostname). Using scratch: $SCRATCH_DIR"
echo "Start time: $JOB_START_HUMAN"

RUN_TAG="${RUN_TAG:-0324}"
SLURM_JOB_TAG="${SLURM_JOB_ID:-manual}"
CHUNK_LOG_FILE="$LOG_DIR/result_merge_${RUN_TAG}_${SLURM_JOB_TAG}.out"
exec > >(tee -a "$CHUNK_LOG_FILE") 2>&1
echo "Merge log file: $CHUNK_LOG_FILE"

print_timing() {
  local status=$1
  local elapsed=$SECONDS
  local h=$((elapsed / 3600))
  local m=$(((elapsed % 3600) / 60))
  local s=$((elapsed % 60))
  local end_human
  end_human="$(date '+%Y-%m-%d %H:%M:%S %Z')"
  local end_epoch
  end_epoch="$(date +%s)"
  echo "End time: $end_human"
  echo "Status: $status"
  echo "Elapsed (SECONDS): ${elapsed}"
  printf 'Elapsed (HH:MM:SS): %02d:%02d:%02d\n' "$h" "$m" "$s"
  echo "Elapsed (date diff): $((end_epoch - JOB_START_EPOCH)) seconds"
}

on_exit() {
  local exit_code=$?
  if [[ $exit_code -eq 0 ]]; then
    print_timing "SUCCESS"
  else
    print_timing "FAILED (exit code ${exit_code})"
  fi
}
trap on_exit EXIT

for f in "$VENV_ACTIVATE" "$MERGE_SCRIPT"; do
  [[ -f "$f" ]] || { echo "Missing required file: $f"; exit 1; }
done

echo "Activating virtual environment at $VENV_ACTIVATE"
source "$VENV_ACTIVATE"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
PY_EXE="$(command -v python3)"
echo "Using Python executable: $PY_EXE"
python3 --version

shopt -s nullglob
TRAIN_INPUTS=( "$CLASSIFICATION_DIR"/output_sig_QCD_classification_${RUN_TAG}_*train_*.h5 )
TEST_INPUTS=( "$CLASSIFICATION_DIR"/output_sig_QCD_classification_${RUN_TAG}_*test_*.h5 )
shopt -u nullglob

if [[ ${#TRAIN_INPUTS[@]} -eq 0 ]]; then
  echo "No train shards found for RUN_TAG=${RUN_TAG} under $CLASSIFICATION_DIR"
  exit 1
fi

if [[ ${#TEST_INPUTS[@]} -eq 0 ]]; then
  echo "No test shards found for RUN_TAG=${RUN_TAG} under $CLASSIFICATION_DIR"
  exit 1
fi

echo "Found ${#TRAIN_INPUTS[@]} train shards"
printf '  %s\n' "${TRAIN_INPUTS[@]}"
echo "Found ${#TEST_INPUTS[@]} test shards"
printf '  %s\n' "${TEST_INPUTS[@]}"

echo "Merging train shards..."
"$PY_EXE" "$MERGE_SCRIPT" \
  -i "${TRAIN_INPUTS[@]}" \
  -o "$MERGED_DIR/merged_train_${RUN_TAG}.h5" \
  --overwrite

echo "Merging test shards..."
"$PY_EXE" "$MERGE_SCRIPT" \
  -i "${TEST_INPUTS[@]}" \
  -o "$MERGED_DIR/merged_test_${RUN_TAG}.h5" \
  --overwrite

echo "Merged outputs written under $MERGED_DIR"

rm -rf "$SCRATCH_DIR"
echo "Scratch directory removed successfully"
