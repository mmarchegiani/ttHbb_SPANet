OPTIONS_FOLDER=/afs/cern.ch/work/m/mmarcheg/ttHbb/ttHbb_SPANet/options_files/ttHbb_semileptonic/classifier/spanet_v3
python scripts/submit_to_condor.py --cfg jobs/config/classification_balance_weights.yaml -of ${OPTIONS_FOLDER}/options_multiple_binary_classifiers_ctag_8M_balance_events.json -l /eos/user/m/mmarcheg/ttHbb/models/spanet_v3/multiple_binary_classifiers_ctag --good-gpus --ngpu 4 --ncpu 12
python scripts/submit_to_condor.py --cfg jobs/config/classification_balance_weights.yaml -of ${OPTIONS_FOLDER}/options_multiple_binary_classifiers_btag_LMH_ctag_8M_balance_events.json -l /eos/user/m/mmarcheg/ttHbb/models/spanet_v3/multiple_binary_classifiers_btag_LMH_ctag --good-gpus --ngpu 4 --ncpu 12
python scripts/submit_to_condor.py --cfg jobs/config/classification_balance_weights.yaml -of ${OPTIONS_FOLDER}/options_multiclassifier_ctag_8M_balance_events.json -l /eos/user/m/mmarcheg/ttHbb/models/spanet_v3/multiclassifier_ctag --good-gpus --ngpu 4 --ncpu 12
python scripts/submit_to_condor.py --cfg jobs/config/classification_balance_weights.yaml -of ${OPTIONS_FOLDER}/options_multiclassifier_btag_LMH_ctag_8M_balance_events.json -l /eos/user/m/mmarcheg/ttHbb/models/spanet_v3/multiclassifier_btag_LMH_ctag --good-gpus --ngpu 4 --ncpu 12
