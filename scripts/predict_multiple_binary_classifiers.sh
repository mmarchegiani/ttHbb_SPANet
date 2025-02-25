MODEL=/eos/user/m/mmarcheg/ttHbb/models/spanet_v2/multiple_binary_classifiers_btag_LMH/spanet_output/version_epoch62
#PREDICTION=/eos/user/m/mmarcheg/ttHbb/predictions/spanet_v2/multiple_binary_classifiers_btag_LMH/spanet_output/version_0/predictions_multiple_binary_classifiers_full_Run2_test_2152028_epoch62.h5
PREDICTION=/eos/user/m/mmarcheg/ttHbb/predictions/spanet_v2/multiple_binary_classifiers_btag_LMH/spanet_output/version_0/predictions_multiple_binary_classifiers_full_Run2_train_8608106_epoch62.h5
VALIDATION=/eos/user/m/mmarcheg/ttHbb/training_datasets/spanet_v2/h5/tthbb_ttbar_with_ctag_one_hot_encoding_full_Run2_train_8608106.h5

# Create output directory if it does not exist
mkdir -p $(dirname $PREDICTION)
python -m spanet.predict $MODEL $PREDICTION -tf $VALIDATION --gpu
