#!/bin/bash
#SBATCH --job-name=recover_h5
#SBATCH --output=logs/recover_%j.out
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --nodelist=rogue01

set -euo pipefail

# Override at submit time if needed, e.g.:
# sbatch --export=ALL,SRC_JOB_ID=196,DEST_DIR=/home/export/$USER slurm/recover_h5_from_scratch.sh
SRC_JOB_ID="${SRC_JOB_ID:-196}"
DEST_DIR="${DEST_DIR:-/home/export/$USER}"
SRC_DIR="/mnt/scratch/$USER/$SRC_JOB_ID"

echo "Running on host: $(hostname)"
echo "Source dir: $SRC_DIR"
echo "Destination dir: $DEST_DIR"

mkdir -p "$DEST_DIR"

shopt -s nullglob
FILES=("$SRC_DIR"/output_sig_0323*.h5)
shopt -u nullglob

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "No files matching output_sig_0323*.h5 found in $SRC_DIR"
  exit 1
fi

cp -v "${FILES[@]}" "$DEST_DIR"/
echo "Copied ${#FILES[@]} file(s) to $DEST_DIR"
