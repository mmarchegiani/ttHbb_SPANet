#!/bin/bash

python scripts/submit_to_condor.py --cfg jobs/config/classification_300epochs.yaml -of options_files/ttHbb_semileptonic/classifier/options_full_Run2_classifier_medium_size_300epochs_btag_LMH.json -l /eos/user/m/mmarcheg/ttHbb/models/classifier_2017_btag_LMH
