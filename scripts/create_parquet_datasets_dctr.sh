# Nested loops over year and datasets to create parquet datasets with spanet inference
if [ $# -lt 1 ]; then
  echo "Error: The input folder should be passed as argument."
  echo "Usage: $0 <FOLDER>"
  exit 1
fi
FOLDER=$1
BASE_FOLDER="/afs/cern.ch/work/m/mmarcheg/ttHbb/ttHbb_SPANet"
subsamples=(tt+B tt+C tt+LF)
datasets_data=(DATA_SingleEle DATA_SingleMuon)
datasets_mc=(
    ttHTobb
    ttHTobb_ttToSemiLep
    TTTo2L2Nu
    ST_s-channel_4f_leptonDecays_s-channel_4f_leptonDecays
    ST_t-channel_antitop_4f_InclusiveDecays_t-channel_antitop_4f_InclusiveDecays
    ST_t-channel_top_4f_InclusiveDecays_t-channel_top_4f_InclusiveDecays
    ST_tW_antitop_5f_NoFullyHadronicDecays_tW_antitop_5f_NoFullyHadronicDecays
    ST_tW_top_5f_NoFullyHadronicDecays_tW_top_5f_NoFullyHadronicDecays
    WJetsToLNu_HT-100To200_100To200
    WJetsToLNu_HT-200To400_200To400
    WJetsToLNu_HT-400To600_400To600
    WJetsToLNu_HT-600To800_600To800
    WJetsToLNu_HT-800To1200_800To1200
    WJetsToLNu_HT-1200To2500_1200To2500
    WJetsToLNu_HT-2500ToInf_2500ToInf
)
datasets_ttsemilep=(TTToSemiLeptonic TTbbSemiLeptonic_4f)
#for year in 2016_PreVFP 2016_PostVFP 2017 2018
for year in 2016_PostVFP 2016_PreVFP 2017 2018
do
    if [ $year == "2016_PreVFP" ]
    then
        eras=(EraB EraC EraD EraE EraF)
    elif [ $year == "2016_PostVFP" ]
    then
        eras=(EraF EraG EraH)
    elif [ $year == "2017" ]
    then
        eras=(EraB EraC EraD EraE EraF)
    elif [ $year == "2018" ]
    then
        eras=(EraA EraB EraC EraD)
    fi
    for dataset in "${datasets_mc[@]}"
    do
        command="python ${BASE_FOLDER}/scripts/dataset/coffea_to_parquet.py --cfg parameters/features_dctr.yaml --cat semilep -i /eos/user/m/mmarcheg/ttHbb/training_datasets/dctr/coffea/${year}/output_all_${year}.coffea -o /eos/user/m/mmarcheg/ttHbb/training_datasets/dctr/parquet/${year}/output_${dataset}_${year}.parquet --ntuples ${FOLDER}/${dataset}_${year}/semilep"
        $command
    done
    for dataset in "${datasets_ttsemilep[@]}"
    do
        # Strip everything after the first underscore
        sample=$(echo $dataset | cut -d'_' -f1)
        # Loop over list of subsamples
        for subs in "${subsamples[@]}"
        do
            echo "Processing $dataset $subs"
            command="python ${BASE_FOLDER}/scripts/dataset/coffea_to_parquet.py --cfg parameters/features_dctr.yaml --cat semilep -i /eos/user/m/mmarcheg/ttHbb/training_datasets/dctr/coffea/${year}/output_all_${year}.coffea -o /eos/user/m/mmarcheg/ttHbb/training_datasets/dctr/parquet/${year}/output_${dataset}_${subs}_${year}.parquet --ntuples ${FOLDER}/${dataset}_${year}/${sample}_${subs}/semilep"
            $command
        done
    done
    for dataset_data in "${datasets_data[@]}"
    do
        for era in "${eras[@]}"
        do
            dataset=${dataset_data}_${year}_${era}
            command="python ${BASE_FOLDER}/scripts/dataset/coffea_to_parquet.py --cfg parameters/features_dctr.yaml --cat semilep -i /eos/user/m/mmarcheg/ttHbb/training_datasets/dctr/coffea/${year}/output_all_${year}.coffea -o /eos/user/m/mmarcheg/ttHbb/training_datasets/dctr/parquet/${year}/output_${dataset}.parquet --ntuples ${FOLDER}/${dataset}/semilep"
            $command
        done
    done
done
