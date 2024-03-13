#!/bin/bash

python scripts/submit_to_condor.py --cfg jobs/config/classification_300epochs.yaml -of options_files/ttHbb_semileptonic/classifier/options_full_Run2_classifier_300epochs.json -l /eos/user/m/mmarcheg/ttHbb/models/classifier_2017_300epochs --checkpoint /eos/user/m/mmarcheg/ttHbb/models/classifier_2017/spanet_output/version_1/checkpoints/last.ckpt
python scripts/submit_to_condor.py --cfg jobs/config/classification_300epochs.yaml -of options_files/ttHbb_semileptonic/classifier/options_full_Run2_classifier_medium_size_300epochs.json -l /eos/user/m/mmarcheg/ttHbb/models/classifier_2017_300epochs --checkpoint /eos/user/m/mmarcheg/ttHbb/models/classifier_2017/spanet_output/version_2/checkpoints/last.ckpt
