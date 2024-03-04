#!/bin/bash
TTHBB_SPANET_DIR=/afs/cern.ch/work/m/mmarcheg/ttHbb/ttHbb_SPANet
NUM_GPU=1

# Create venv in local job dir
python -m venv myenv --system-site-packages
source myenv/bin/activate

# Install SPANet in virtual environment
cd $TTHBB_SPANET_DIR
pip install -e .

# Launch training
python -m spanet.train \
       --options_file $1 \
       --log-dir $2\
       --time_limit 02:00:00:00\
       --gpus $NUM_GPU