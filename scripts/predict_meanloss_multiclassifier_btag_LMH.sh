MODEL=/eos/home-m/mmarcheg/ttHbb/models/meanloss_multiclassifier_btag_LMH/spanet_output/version_0
PREDICTION=/eos/user/m/mmarcheg/ttHbb/predictions/meanloss_multiclassifier_btag_LMH/spanet_output/version_0/predictions_tthbb_ttbar_with_ttbb_merging_ttbb_reweighed_full_Run2_test_1994982_epoch285.h5
VALIDATION=/eos/user/m/mmarcheg/ttHbb/training_datasets/classifier/multiclassifier_full_Run2_ttbb_reweighed/tthbb_ttbar_with_ttbb_merging_ttbb_reweighed_full_Run2_test_1994982.h5

# Create output directory if it does not exist
mkdir -p $(dirname $PREDICTION)
python -m spanet.predict $MODEL $PREDICTION -tf $VALIDATION --gpu
