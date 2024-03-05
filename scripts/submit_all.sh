#!/bin/bash

python scripts/submit_to_condor.py --cfg jobs/config/jet_assignment.yaml -of options_files/ttHbb_semileptonic/options_full_Run2.json -l /eos/user/m/mmarcheg/ttHbb/models/jet_assignment_full_Run2/ --checkpoint /eos/user/m/mmarcheg/ttHbb/models/jet_assignment_full_Run2/spanet_output/version_2/checkpoints/last.ckpt
python scripts/submit_to_condor.py --cfg jobs/config/jet_assignment.yaml -of options_files/ttHbb_semileptonic/options_full_Run2_btag_L.json -l /eos/user/m/mmarcheg/ttHbb/models/jet_assignment_full_Run2_btag_L
python scripts/submit_to_condor.py --cfg jobs/config/jet_assignment.yaml -of options_files/ttHbb_semileptonic/options_full_Run2_btag_M.json -l /eos/user/m/mmarcheg/ttHbb/models/jet_assignment_full_Run2_btag_M
python scripts/submit_to_condor.py --cfg jobs/config/jet_assignment.yaml -of options_files/ttHbb_semileptonic/options_full_Run2_btag_H.json -l /eos/user/m/mmarcheg/ttHbb/models/jet_assignment_full_Run2_btag_H
