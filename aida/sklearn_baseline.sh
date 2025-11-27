#!/bin/bash

# Configuration
data_dir_root="./data/fit-medium-table"

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

# Model types to test
models=("logistic" "randomforest")

# Function to run ML baseline
run_ml_baseline() {
    local dataset=$1
    local model=$2
    local data_dir="${data_dir_root}/${dataset}"

    python -m cmds.sklearn_baseline \
        --data_dir "${data_dir}" \
        --method "${model}"
}

# Loop through all datasets
for dataset in "${data_list[@]}"; do

    # Test ML models
    for model in "${models[@]}"; do
        run_ml_baseline "${dataset}" "${model}"
    done

    echo ""
done

echo "All baseline tests completed!"