# DFS-DNN#!/bin/bash
# Base model testing script

# Configuration
data_dir_root="./data/dfs-fs-table"

# List of datasets to test
data_list=(
    "avito-user-clicks"
    # "avito-ad-ctr"
    "event-user-repeat"
    "event-user-attendance"
    "ratebeer-beer-positive"
    "ratebeer-place-positive"
    "ratebeer-user-active"
    "trial-site-success"
    "trial-study-outcome"
    "hm-item-sales"
    "hm-user-churn"
)



models=("MLP")

# Function to run DNN baseline
run_dnn_baseline() {
    local dataset=$1
    local model=$2
    local data_dir="${data_dir_root}/${dataset}"

    python -m cmds.dnn_baseline_table_data \
        --data_dir "${data_dir}" \
        --model "${model}"
}


# Loop through all datasets
for dataset in "${data_list[@]}"; do

    # Test DNN models
    for model in "${models[@]}"; do
        run_dnn_baseline "${dataset}" "${model}"
    done
    echo ""
done

echo "All baseline tests completed!"