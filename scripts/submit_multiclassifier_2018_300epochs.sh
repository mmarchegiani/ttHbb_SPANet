#!/bin/bash

python scripts/submit_to_condor.py --cfg jobs/config/classification_300epochs.yaml -of options_files/ttHbb_semileptonic/classifier/options_multiclassifier_2018_300epochs.json -l /eos/user/m/mmarcheg/ttHbb/models/multiclassifier_2018_300epochs
python scripts/submit_to_condor.py --cfg jobs/config/classification_300epochs.yaml -of options_files/ttHbb_semileptonic/classifier/options_multiclassifier_2018_300epochs_btag_LMH.json -l /eos/user/m/mmarcheg/ttHbb/models/multiclassifier_2018_300epochs_btag_LMH
