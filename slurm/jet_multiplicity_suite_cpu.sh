#!/bin/bash
#SBATCH --job-name=jetMultSuite
#SBATCH --output=logs/jet_mult_suite_%j.out
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00

set -euo pipefail

PROJECT_DIR="/home/export/sdurgut/scratch/ttHbb_SPANet"
PY="/home/export/sdurgut/scratch/ttHbb_SPANet/envs/notebook_env/bin/python"
SCRIPT="${PROJECT_DIR}/scripts/dataset/jet_multiplicity_suite.py"

# Your parquet/H5 locations
PARQUET_BASE="/home/export/sdurgut/scratch/coffea_to_parquet_output"
H5_BASE="/home/export/sdurgut/scratch/parquet_to_h5_output/classification/with_MASK"

# Tag used in H5 shard names
RUN_TAG="${RUN_TAG:-0324}"

# EOS destination (optional). Example:
#   export EOS_DIR="root://eosuser.cern.ch//eos/user/s/sdurgut/www/ttH(bb)/jet_mult/jet_mult_${RUN_TAG}_$(date +%Y%m%d_%H%M%S)"
EOS_DIR="${EOS_DIR:-}"

mkdir -p "${PROJECT_DIR}/logs"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "RUN_TAG=${RUN_TAG}"
echo "PARQUET_BASE=${PARQUET_BASE}"
echo "H5_BASE=${H5_BASE}"
echo "Python: ${PY}"

OUT_DIR="${PROJECT_DIR}/outputs/jet_mult_${RUN_TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT_DIR}"
echo "OUT_DIR=${OUT_DIR}"

"${PY}" -c "import awkward, pyarrow, h5py; print('deps ok', awkward.__version__)" || true

# Run suite
CMD=(
  "${PY}" "${SCRIPT}"
  --run_tag "${RUN_TAG}"
  --parquet_base "${PARQUET_BASE}"
  --h5_base "${H5_BASE}"
  --out_dir "${OUT_DIR}"
)

if [[ -n "${EOS_DIR}" ]]; then
  CMD+=( --eos_dir "${EOS_DIR}" )
else
  CMD+=( --skip_eos )
fi

echo "Running:"
printf '  %q' "${CMD[@]}"; echo
"${CMD[@]}"

echo "End: $(date)"
echo "Done."

