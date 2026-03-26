#!/bin/bash
#SBATCH --job-name=get_h5_cpu
#SBATCH --output=logs/result_%j.out
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=4G

set -euo pipefail

SECONDS=0
JOB_START_EPOCH="$(date +%s)"
JOB_START_HUMAN="$(date '+%Y-%m-%d %H:%M:%S %Z')"

PROJECT_DIR="/home/export/sdurgut/scratch/ttHbb_SPANet"
OUTPUT_DIR="/home/export/sdurgut/scratch/parquet_to_h5_output/classification"
mkdir -p "$OUTPUT_DIR"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
PARQUET_BASE="/home/export/sdurgut/coffea_to_parquet_output"
VENV_ACTIVATE="$PROJECT_DIR/SPANet/spanet_env_cpu/bin/activate"
SCRATCH_DIR="/mnt/scratch/$USER/$SLURM_JOB_ID"
echo "Job running on $(hostname). Using scratch: $SCRATCH_DIR"
mkdir -p "$SCRATCH_DIR"

CFG_SRC="$PROJECT_DIR/parameters/h5_params/ttHbb_fully_hadronic/features_h5_classification.yaml"
PY_SRC="$PROJECT_DIR/scripts/dataset/parquet_to_h5.py"
OUTPUT_PREFIX="output_sig_QCD_classification_0324"
CHUNK="${CHUNK:-${1:-}}"

for f in "$CFG_SRC" "$PY_SRC" "$VENV_ACTIVATE"; do
  [[ -f "$f" ]] || { echo "Missing required file: $f"; exit 1; }
done
echo "Inputs validated successfully"

cp "$CFG_SRC" "$SCRATCH_DIR/features_h5_classification.yaml"
cp "$PY_SRC" "$SCRATCH_DIR/parquet_to_h5.py"
echo "Inputs copied to scratch directory successfully"

cd "$SCRATCH_DIR"
echo "Activating virtual environment at $VENV_ACTIVATE"
source "$VENV_ACTIVATE"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
PY_EXE="$(command -v python3)"
echo "Using Python executable: $PY_EXE"
python3 --version

usage() {
  echo "Usage:"
  echo "  sbatch --export=ALL,CHUNK=<chunk_name> slurm/get_h5_cpu_rogue.sh"
  echo "Valid chunk_name values:"
  echo "  signal_only"
  echo "  qcd_ht200_600"
  echo "  qcd_ht600_1000"
  echo "  qcd_ht1000_1500"
  echo "  qcd_ht1500_2000"
  echo "  qcd_ht2000"
}

collect_inputs() {
  local -n out_arr=$1
  shift
  out_arr=()
  shopt -s nullglob
  for pattern in "$@"; do
    for f in "$PARQUET_BASE"/$pattern; do
      out_arr+=("$f")
    done
  done
  shopt -u nullglob
}

run_chunk() {
  local chunk_name=$1
  shift
  local patterns=("$@")
  local inputs=()
  collect_inputs inputs "${patterns[@]}"

  if [[ ${#inputs[@]} -eq 0 ]]; then
    echo "No parquet files found for chunk ${chunk_name}"
    exit 1
  fi

  for f in "${inputs[@]}"; do
    [[ -f "$f" ]] || { echo "Missing parquet file: $f"; exit 1; }
  done

  echo "Running chunk ${chunk_name} with ${#inputs[@]} files"
  "$PY_EXE" "$SCRATCH_DIR/parquet_to_h5.py" \
    --cfg "$SCRATCH_DIR/features_h5_classification.yaml" \
    -i "${inputs[@]}" \
    -o "$SCRATCH_DIR/${OUTPUT_PREFIX}_${chunk_name}.h5"
}

if [[ -z "$CHUNK" ]]; then
  echo "CHUNK is required for one-by-one debugging."
  usage
  exit 1
fi

SLURM_JOB_TAG="${SLURM_JOB_ID:-manual}"
CHUNK_LOG_FILE="$LOG_DIR/result_${CHUNK}_${SLURM_JOB_TAG}.out"
exec > >(tee -a "$CHUNK_LOG_FILE") 2>&1
echo "Chunk log file: $CHUNK_LOG_FILE"
echo "Start time: $JOB_START_HUMAN"

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

case "$CHUNK" in
  signal_only)
    run_chunk signal_only \
      "output_TTH_Hto2B_2022_postEE.parquet" \
      "output_TTH_Hto2B_2022_preEE.parquet" \
      "output_TTH_Hto2B_2023_postBPix.parquet" \
      "output_TTH_Hto2B_2023_preBPix.parquet"
    ;;
  qcd_ht200_600)
    run_chunk qcd_ht200_600 \
      "output_QCD-4Jets_HT-200to400_*.parquet" \
      "output_QCD-4Jets_HT-400to600_*.parquet"
    ;;
  qcd_ht600_1000)
    run_chunk qcd_ht600_1000 \
      "output_QCD-4Jets_HT-600to800_*.parquet" \
      "output_QCD-4Jets_HT-800to1000_*.parquet"
    ;;
  qcd_ht1000_1500)
    run_chunk qcd_ht1000_1500 \
      "output_QCD-4Jets_HT-1000to1200_*.parquet" \
      "output_QCD-4Jets_HT-1200to1500_*.parquet"
    ;;
  qcd_ht1500_2000)
    run_chunk qcd_ht1500_2000 \
      "output_QCD-4Jets_HT-1500to2000_*.parquet"
    ;;
  qcd_ht2000)
    run_chunk qcd_ht2000 \
      "output_QCD-4Jets_HT-2000_*.parquet"
    ;;
  *)
    echo "Invalid CHUNK: $CHUNK"
    usage
    exit 1
    ;;
esac

echo "Chunk ${CHUNK} completed successfully"

shopt -s nullglob
OUTPUT_FILES=("$SCRATCH_DIR"/${OUTPUT_PREFIX}_*.h5)
echo "Output files found: ${OUTPUT_FILES[@]}"
shopt -u nullglob

if [[ ${#OUTPUT_FILES[@]} -eq 0 ]]; then
  echo "No output files matching ${OUTPUT_PREFIX}_*.h5 were created."
  exit 1
fi

cp "${OUTPUT_FILES[@]}" "$OUTPUT_DIR/"
echo "Copied ${#OUTPUT_FILES[@]} file(s) to $OUTPUT_DIR successfully"

rm -rf "$SCRATCH_DIR"
echo "Scratch directory removed successfully"