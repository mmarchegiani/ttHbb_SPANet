#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
NUM_GPU=1
cd /opt/SPANet
python -m spanet.train -of  /afs/cern.ch/work/m/mmarcheg/ttHbb/SPANet/options_files/full_hadronic_ttbar/example.json --time_limit 00:00:01:00 --gpus $NUM_GPU

