#!/bin/bash
SPANET_DIR=/afs/cern.ch/work/m/mmarcheg/ttHbb/dev/SPANet
TTHBB_SPANET_DIR=/afs/cern.ch/work/m/mmarcheg/ttHbb/ttHbb_SPANet
NUM_GPU=1

# Create venv in local job dir
python -m venv myenv --system-site-packages
source myenv/bin/activate

# Install SPANet in virtual environment
cd $SPANET_DIR
pip install -e .

# Install ttHbb_SPANet in virtual environment
cd $TTHBB_SPANET_DIR
pip install -e .

# Install MDMM loss package
python -m pip install 'mdmm @ git+https://github.com/the-moliver/mdmm@master'

# Launch training
if [ $# -eq 2 ]; then
    python -m spanet.train \
           --options_file $1 \
           --log_dir $2\
           --time_limit 07:00:00:00\
           --gpus $NUM_GPU
elif [ $# -eq 3 ]; then
    python -m spanet.train \
           --options_file $1 \
           --log_dir $2\
           --checkpoint $3\
           --time_limit 07:00:00:00\
           --gpus $NUM_GPU
fi
