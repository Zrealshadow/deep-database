#!/bin/bash

# AIDA Model Ablation Study - Fusion and Relation Module Deactivation
# Usage: ./run_aida_ablation.sh [base_encoder]

# Note: Not using 'set -e' to allow continuation after CUDA OOM errors

BASE_ENCODER=${1:-""}

# Define datasets with their configurations
# Format: "db_name|tf_cache_dir|task_name"
CLASSIFICATION_DATASETS=(
    "hm|data/hm-tensor-frame|user-churn"
    "event|data/rel-event-tensor-frame|user-repeat"
    # "trial|data/rel-trial-tensor-frame|study-outcome"
    "avito|data/rel-avito-tensor-frame|user-clicks"
    # "ratebeer|data/ratebeer-tensor-frame|user-active"
)

REGRESSION_DATASETS=(
    "hm|data/hm-tensor-frame|item-sales"
    "event|data/rel-event-tensor-frame|user-attendance"
    "avito|data/rel-avito-tensor-frame|ad-ctr"
    # "trial|data/rel-trial-tensor-frame|site-success"
    # "ratebeer|data/ratebeer-tensor-frame|beer-positive"
)

# Combine all datasets
ALL_DATASETS=("${CLASSIFICATION_DATASETS[@]}" "${REGRESSION_DATASETS[@]}")

# Define ablation configurations
# 0: baseline (both modules active)
# 1: deactivate fusion module only
# 2: deactivate relation module only
ABLATION_CONFIGS=(
    "baseline|"
    "no_fusion|--deactivate_fusion_module"
    "no_relation|--deactivate_relation_module"
)

# Add base encoder if provided
COMMON_ARGS=""
if [ -n "$BASE_ENCODER" ]; then
    COMMON_ARGS="--base_encoder $BASE_ENCODER"
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

    # Loop through all ablation configurations
    for ablation_config in "${ABLATION_CONFIGS[@]}"; do
        # Parse ablation configuration
        IFS='|' read -r ABLATION_NAME ABLATION_FLAGS <<< "$ablation_config"

        echo "=========================================="
        echo "Running Ablation Experiment:"
        echo "  Database: $DB_NAME"
        echo "  Task: $TASK_NAME"
        echo "  Configuration: $ABLATION_NAME"
        echo "  Flags: $ABLATION_FLAGS"
        echo "=========================================="

        # Run the experiment, continue even if it fails (e.g., CUDA OOM)
        python -m aida.aida_run \
            --db_name "$DB_NAME" \
            --tf_cache_dir "$TF_CACHE_DIR" \
            --task_name "$TASK_NAME" \
            --early_stop_threshold $EARLY_STOP \
            $ABLATION_FLAGS \
            $COMMON_ARGS || echo "FAILED: DB=$DB_NAME, Task=$TASK_NAME, Config=$ABLATION_NAME"
    done
done
