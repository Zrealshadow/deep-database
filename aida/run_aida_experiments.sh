#!/bin/bash

# AIDA Model Experiments with Different Base Encoders
# This script runs experiments across multiple datasets and tasks with different encoder configurations

set -e  # Exit on error

# Define base encoders to test
ENCODERS=("mlp" "tabm" "dfm" "resnet" "fttrans")

# Define datasets with their configurations
# Format: "db_name|tf_cache_dir|task_name"
CLASSIFICATION_DATASETS=(
    "hm|data/hm-tensor-frame|user-churn"
    "event|data/rel-event-tensor-frame|user-repeat"
    "trial|data/rel-trial-tensor-frame|study-outcome"
    "avito|data/rel-avito-tensor-frame|user-clicks"
    "ratebeer|data/ratebeer-tensor-frame|place-positive"
    "ratebeer|data/ratebeer-tensor-frame|user-active"
)

REGRESSION_DATASETS=(
    "hm|data/hm-tensor-frame|item-sales"
    "event|data/rel-event-tensor-frame|user-attendance"
    "avito|data/rel-avito-tensor-frame|ad-ctr"
    "trial|data/rel-trial-tensor-frame|site-success"
    "ratebeer|data/ratebeer-tensor-frame|beer-positive"
)

# Optional: Add common arguments here
COMMON_ARGS="--no_need_test"

echo "=========================================="
echo "Starting AIDA Experiments"
echo "=========================================="
echo ""

# Process Classification Datasets
echo "=========================================="
echo "Classification Tasks"
echo "=========================================="
echo ""

for dataset_config in "${CLASSIFICATION_DATASETS[@]}"; do
    # Parse dataset configuration
    IFS='|' read -r db_name tf_cache_dir task_name <<< "$dataset_config"

    # Loop through all encoders
    for encoder in "${ENCODERS[@]}"; do

        python -m aida.aida_run \
            --db_name "$db_name" \
            --tf_cache_dir "$tf_cache_dir" \
            --task_name "$task_name" \
            --base_encoder "$encoder" \
            $COMMON_ARGS 
    done
    echo ""
done

echo ""
echo "=========================================="
echo "Regression Tasks"
echo "=========================================="
echo ""

# Process Regression Datasets
for dataset_config in "${REGRESSION_DATASETS[@]}"; do
    # Parse dataset configuration
    IFS='|' read -r db_name tf_cache_dir task_name <<< "$dataset_config"

    # Loop through all encoders
    for encoder in "${ENCODERS[@]}"; do
        python -m aida.aida_run \
            --db_name "$db_name" \
            --tf_cache_dir "$tf_cache_dir" \
            --task_name "$task_name" \
            --base_encoder "$encoder" \
            $COMMON_ARGS 
    done
    echo ""
done

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
