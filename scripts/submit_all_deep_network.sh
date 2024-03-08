#!/bin/bash

python scripts/submit_to_condor.py --cfg jobs/config/jet_assignment_deep_network.yaml -of options_files/ttHbb_semileptonic/jet_assignment/options_full_Run2_deep_network.json -l /eos/user/m/mmarcheg/ttHbb/models/jet_assignment_full_Run2_deep_network
python scripts/submit_to_condor.py --cfg jobs/config/jet_assignment_deep_network.yaml -of options_files/ttHbb_semileptonic/jet_assignment/options_full_Run2_btag_LMH_deep_network.json -l /eos/user/m/mmarcheg/ttHbb/models/jet_assignment_full_Run2_btag_LMH_deep_network
python scripts/submit_to_condor.py --cfg jobs/config/jet_assignment_deep_network.yaml -of options_files/ttHbb_semileptonic/jet_assignment/options_full_Run2_btag_L_deep_network.json -l /eos/user/m/mmarcheg/ttHbb/models/jet_assignment_full_Run2_btag_L_deep_network
python scripts/submit_to_condor.py --cfg jobs/config/jet_assignment_deep_network.yaml -of options_files/ttHbb_semileptonic/jet_assignment/options_full_Run2_btag_M_deep_network.json -l /eos/user/m/mmarcheg/ttHbb/models/jet_assignment_full_Run2_btag_M_deep_network
python scripts/submit_to_condor.py --cfg jobs/config/jet_assignment_deep_network.yaml -of options_files/ttHbb_semileptonic/jet_assignment/options_full_Run2_btag_H_deep_network.json -l /eos/user/m/mmarcheg/ttHbb/models/jet_assignment_full_Run2_btag_H_deep_network
