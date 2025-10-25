# TabPFN baseline



# DFS-DNN#!/bin/bash
# Base model testing script

# Configuration
data_dir_root="./data/dfs-fs-table"

# List of datasets to test
data_list=(
    "avito-user-clicks"
    "avito-ad-ctr"
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




# Function to run DNN baseline
run_dnn_baseline() {
    local dataset=$1
    local data_dir="${data_dir_root}/${dataset}"
    python -m tabPFN.tabpfn_baseline \
        --data_dir "${data_dir}" \
        --verbose
}


# Loop through all datasets
for dataset in "${data_list[@]}"; do
    run_dnn_baseline "${dataset}" 
done

echo "All baseline tests completed!"