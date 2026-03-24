#!/bin/bash
#SBATCH --job-name=get_h5_cpu
#SBATCH --output=logs/result_%j.out
#SBATCH --partition=work        # Standard CPU partition
#SBATCH --nodes=1                  # Run on a single node
#SBATCH --ntasks=1                 # Single task
#SBATCH --cpus-per-task=4          # Number of CPU cores
#SBATCH --mem-per-cpu=1G           # 4 CPUs x 1G = 4G total (<= 4G/CPU limit)

set -euo pipefail

# 1. SETUP: Define a unique scratch directory
PROJECT_DIR="/home/export/sdurgut/scratch/ttHbb_SPANet"
PARQUET_BASE="/home/export/sdurgut/training_files"
VENV_ACTIVATE="$PROJECT_DIR/SPANet/spanet_env_container_backup/bin/activate"
SCRATCH_DIR="/mnt/scratch/$USER/$SLURM_JOB_ID"
echo "Job running on $(hostname). Using scratch: $SCRATCH_DIR"
mkdir -p "$SCRATCH_DIR"

# 1b. PREFLIGHT: Validate inputs before heavy work starts
CFG_SRC="$PROJECT_DIR/parameters/ttHbb_fully_hadronic/features_h5_jet_assignment.yaml"
PY_SRC="$PROJECT_DIR/scripts/dataset/parquet_to_h5.py"
INPUTS=(
  "$PARQUET_BASE/output_TTH_Hto2B_2022_postEE.parquet"
  "$PARQUET_BASE/output_TTH_Hto2B_2022_preEE.parquet"
  "$PARQUET_BASE/output_TTH_Hto2B_2023_postBPix.parquet"
  "$PARQUET_BASE/output_TTH_Hto2B_2023_preBPix.parquet"
)

for f in "$CFG_SRC" "$PY_SRC" "$VENV_ACTIVATE" "${INPUTS[@]}"; do
  [[ -f "$f" ]] || { echo "Missing required file: $f"; exit 1; }
done

# 2. STAGE IN: Copy only light inputs (avoid copying large parquet files)
cp "$CFG_SRC" "$SCRATCH_DIR/features_h5_jet_assignment.yaml"
cp "$PY_SRC" "$SCRATCH_DIR/parquet_to_h5.py"

# 3. RUN: Execute code using the local fast data
cd "$SCRATCH_DIR"
echo "Activating virtual environment at $VENV_ACTIVATE"
source "$VENV_ACTIVATE"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
PY_EXE="$(command -v python3)"
echo "Using Python executable: $PY_EXE"
python3 --version

if ! "$PY_EXE" - <<'PY'
import traceback
try:
    import numba
    print(f"numba import OK: {numba.__version__}")
except Exception:
    print("Cannot import 'numba' in this runtime.")
    traceback.print_exc()
    raise SystemExit(1)
PY
then
  echo "numba import check failed."
  exit 1
fi

"$PY_EXE" "$SCRATCH_DIR/parquet_to_h5.py" \
  --cfg "$SCRATCH_DIR/features_h5_jet_assignment.yaml" \
  -i "${INPUTS[@]}" \
  -o "$SCRATCH_DIR/output_sig_0323.h5"

# 4. STAGE OUT: Copy results from Fast Scratch -> project directory
shopt -s nullglob
OUTPUT_FILES=("$SCRATCH_DIR"/output_sig_0323*.h5)
shopt -u nullglob

if [[ ${#OUTPUT_FILES[@]} -eq 0 ]]; then
  echo "No output files matching output_sig_0323*.h5 were created."
  exit 1
fi

cp "${OUTPUT_FILES[@]}" "$PROJECT_DIR/"
echo "Copied ${#OUTPUT_FILES[@]} file(s) to $PROJECT_DIR"

# 5. CLEANUP: Remove temp files
rm -rf "$SCRATCH_DIR"