MODEL=/eos/user/m/mmarcheg/ttHbb/models/spanet_v2/multiclassifier_btag_LMH/spanet_output/version_1
PREDICTION=/eos/user/m/mmarcheg/ttHbb/predictions/spanet_v2/multiclassifier_btag_LMH/spanet_output/version_1/predictions_multiclassifier_full_Run2_test_2152028.h5
VALIDATION=/eos/user/m/mmarcheg/ttHbb/training_datasets/spanet_v2/h5/tthbb_ttbar_with_ctag_one_hot_encoding_full_Run2_test_2152028.h5

# Create output directory if it does not exist
mkdir -p $(dirname $PREDICTION)
python -m spanet.predict $MODEL $PREDICTION -tf $VALIDATION --gpu
