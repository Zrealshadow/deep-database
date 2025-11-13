#!/bin/bash

# AIDA Model Experiments - Neighbor Size and Sampling Strategy Validation
# Usage: ./run_aida_neighbor_size.sh [base_encoder]

set -e

BASE_ENCODER=${1:-""}

# Define datasets with their configurations
# Format: "db_name|tf_cache_dir|task_name"
CLASSIFICATION_DATASETS=(
    "hm|data/hm-tensor-frame|user-churn"
    "event|data/rel-event-tensor-frame|user-repeat"
    "trial|data/rel-trial-tensor-frame|study-outcome"
    "avito|data/rel-avito-tensor-frame|user-clicks"
    "ratebeer|data/ratebeer-tensor-frame|user-active"
)

REGRESSION_DATASETS=(
    "hm|data/hm-tensor-frame|item-sales"
    "event|data/rel-event-tensor-frame|user-attendance"
    "avito|data/rel-avito-tensor-frame|ad-ctr"
    "trial|data/rel-trial-tensor-frame|site-success"
    "ratebeer|data/ratebeer-tensor-frame|beer-positive"
)

# Combine all datasets
ALL_DATASETS=("${CLASSIFICATION_DATASETS[@]}" "${REGRESSION_DATASETS[@]}")

# Define neighbor sizes to test
NEIGHBORS_1HOP=(
    "8"
    "16"
    "32"
    "64"
    "128"
    "256"
    "512"
)

NEIGHBORS_2HOP=(
    "8 8"
    "16 16"
    "32 32"
    "64 64"
    "128 128"
    "256 256"
)

NEIGHBORS_3HOP=(
    "8 8 8"
    "16 16 16"
    "32 32 32"
    "64 64 64"
)

NEIGHBORS_4HOP=(
    "8 8 8 8"
    "16 16 16 16"
    "32 32 32 32"
)

# Combine all neighbor configurations
ALL_NEIGHBORS=("${NEIGHBORS_1HOP[@]}" "${NEIGHBORS_2HOP[@]}" "${NEIGHBORS_3HOP[@]}" "${NEIGHBORS_4HOP[@]}")

# Define sampling strategies to test
SAMPLE_STRATEGIES=("last" "uniform")

# Common arguments
COMMON_ARGS="--no_need_test"

# Add base encoder if provided
if [ -n "$BASE_ENCODER" ]; then
    COMMON_ARGS="$COMMON_ARGS --base_encoder $BASE_ENCODER"
fi

# Function to check if dataset is classification
is_classification() {
    local config=$1
    for cls_config in "${CLASSIFICATION_DATASETS[@]}"; do
        if [ "$config" = "$cls_config" ]; then
            return 0
        fi
    done
    return 1
}

# Loop through all datasets
for dataset_config in "${ALL_DATASETS[@]}"; do
    # Parse dataset configuration
    IFS='|' read -r DB_NAME TF_CACHE_DIR TASK_NAME <<< "$dataset_config"

    # Set early_stop_threshold based on task type
    if is_classification "$dataset_config"; then
        EARLY_STOP=5
    else
        EARLY_STOP=10
    fi

    # Loop through all sampling strategies
    for strategy in "${SAMPLE_STRATEGIES[@]}"; do
        # Loop through all neighbor configurations
        for neighbors in "${ALL_NEIGHBORS[@]}"; do
            # Run the experiment
            python -m aida.aida_run \
                --db_name "$DB_NAME" \
                --tf_cache_dir "$TF_CACHE_DIR" \
                --task_name "$TASK_NAME" \
                --num_neighbors $neighbors \
                --sample_strategy "$strategy" \
                --early_stop_threshold $EARLY_STOP \
                $COMMON_ARGS
        done
    done
done
