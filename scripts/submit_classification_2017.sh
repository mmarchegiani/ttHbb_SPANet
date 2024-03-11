#!/bin/bash

python scripts/submit_to_condor.py --cfg jobs/config/classification.yaml -of options_files/ttHbb_semileptonic/classifier/options_full_Run2_classifier.json -l /eos/user/m/mmarcheg/ttHbb/models/classifier_2017
python scripts/submit_to_condor.py --cfg jobs/config/classification.yaml -of options_files/ttHbb_semileptonic/classifier/options_full_Run2_classifier_medium_size.json -l /eos/user/m/mmarcheg/ttHbb/models/classifier_2017
