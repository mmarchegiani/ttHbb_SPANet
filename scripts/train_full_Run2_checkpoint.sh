#!/bin/bash

NUM_GPU=1
nohup python -m spanet.train \
         -of /afs/cern.ch/work/m/mmarcheg/ttHbb/ttHbb_SPANet/options_files/ttHbb_semileptonic/options_full_Run2.json \
         --log_dir /eos/user/m/mmarcheg/ttHbb/models/jet_assignment_full_Run2 \
         --checkpoint /eos/user/m/mmarcheg/ttHbb/models/jet_assignment_full_Run2/spanet_output/version_1/checkpoints/last.ckpt \
         --time_limit 00:24:00:00 --gpus $NUM_GPU > /afs/cern.ch/work/m/mmarcheg/ttHbb/ttHbb_SPANet/logs/training_jet_assignment_full_Run2_checkpoint.log 2>&1 &
