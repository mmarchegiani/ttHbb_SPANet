#!/bin/bash

python scripts/submit_to_condor.py --cfg jobs/config/test_validation_loss.yaml -of /afs/cern.ch/work/m/mmarcheg/ttHbb/ttHbb_SPANet/options_files/ttHbb_semileptonic/test/options_multiclassifier_2018_300epochs.json -l /afs/cern.ch/work/m/mmarcheg/ttHbb/ttHbb_SPANet/test_validation_loss
