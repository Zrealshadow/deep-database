#!/bin/bash
# ARDA baseline testing script (Lasso for regression, Random Forest for classification)

# Configuration
data_dir_root="./data/arda-table"

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

# Function to run ARDA baseline
run_arda_baseline() {
    local dataset=$1
    local data_dir="${data_dir_root}/${dataset}"

    echo "=========================================="
    echo "Running ARDA baseline on ${dataset}"
    echo "=========================================="

    python -m cmds.arda_baseline \
        --data_dir "${data_dir}"
}

# Main execution
echo "Starting ARDA baseline testing..."
echo ""

# Loop through all datasets
for dataset in "${data_list[@]}"; do
    run_arda_baseline "${dataset}"
    echo ""
done

echo "All ARDA baseline tests completed!"