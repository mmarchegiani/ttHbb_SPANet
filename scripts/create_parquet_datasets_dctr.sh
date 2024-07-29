# Nested loops over year and datasets ttHTobb and ttHTobb_ttToSemiLep to create parquet datasets with improved matching
#for year in 2016_PreVFP 2016_PostVFP 2017 2018
subsamples=(tt+B tt+C tt+LF)
datasets_data=(DATA_SingleEle DATA_SingleMuon)
#for year in 2018 2017 2016_PostVFP 2016_PreVFP
for year in 2017 2016_PostVFP 2016_PreVFP
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
    for dataset in ttHTobb TTTo2L2Nu
    do
        python scripts/dataset/coffea_to_parquet.py --cfg parameters/features_dctr.yaml --cat semilep -i /eos/user/m/mmarcheg/ttHbb/training_datasets/dctr/coffea/${year}/output_all.coffea -o /eos/user/m/mmarcheg/ttHbb/training_datasets/dctr/parquet/${year}/output_${dataset}_${year}.parquet --ntuples /eos/user/m/mmarcheg/ttHbb/training_datasets/dctr/parquet/parton_matching_old_matching_with_event_features_22_07_24/${dataset}_${year}/semilep
    done
    for dataset in TTToSemiLeptonic TTbbSemiLeptonic_4f
    do
        # Strip everything after the first underscore
        sample=$(echo $dataset | cut -d'_' -f1)
        # Loop over list of subsamples
        for subs in "${subsamples[@]}"
        do
            echo "Processing $dataset $subs"
            python scripts/dataset/coffea_to_parquet.py --cfg parameters/features_dctr.yaml --cat semilep -i /eos/user/m/mmarcheg/ttHbb/training_datasets/dctr/coffea/${year}/output_all.coffea -o /eos/user/m/mmarcheg/ttHbb/training_datasets/dctr/parquet/${year}/output_${dataset}_${subs}_${year}.parquet --ntuples /eos/user/m/mmarcheg/ttHbb/training_datasets/dctr/parquet/parton_matching_old_matching_with_event_features_22_07_24/${dataset}_${year}/${sample}_${subs}/semilep
        done
    done
    for dataset_data in "${datasets_data[@]}"
    do
        for era in "${eras[@]}"
        do
            dataset=${dataset_data}_${year}_${era}
            python scripts/dataset/coffea_to_parquet.py --cfg parameters/features_dctr.yaml --cat semilep -i /eos/user/m/mmarcheg/ttHbb/training_datasets/dctr/coffea/${year}/output_all.coffea -o /eos/user/m/mmarcheg/ttHbb/training_datasets/dctr/parquet/${year}/output_${dataset}.parquet --ntuples /eos/user/m/mmarcheg/ttHbb/training_datasets/dctr/parquet/parton_matching_old_matching_with_event_features_22_07_24/${dataset}/semilep
        done
    done
done
