#!/bin/bash
# Base model testing script

# Configuration
data_dir_root="./data/flatten-table"

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
models=("MLP" "FTTrans" "ResNet")

# ML baseline methods
ml_methods=("lgb" "catboost")

# Function to run DNN baseline
run_dnn_baseline() {
    local dataset=$1
    local model=$2
    local data_dir="${data_dir_root}/${dataset}"

    python -m cmds.dnn_baseline_table_data \
        --data_dir "${data_dir}" \
        --model "${model}"
}

# Function to run ML baseline
run_ml_baseline() {
    local dataset=$1
    local method=$2
    local data_dir="${data_dir_root}/${dataset}"

    python -m cmds.ml_baseline \
        --data_dir "${data_dir}" \
        --method "${method}"
}

# Main execution
echo "Starting baseline model testing..."
echo ""

# Loop through all datasets
for dataset in "${data_list[@]}"; do
    echo "======================================"
    echo "Testing dataset: ${dataset}"
    echo "======================================"

    # Test DNN models
    for model in "${models[@]}"; do
        run_dnn_baseline "${dataset}" "${model}"
    done

    # Test ML models
    for method in "${ml_methods[@]}"; do
        run_ml_baseline "${dataset}" "${method}"
    done

    echo ""
done

echo "All baseline tests completed!"