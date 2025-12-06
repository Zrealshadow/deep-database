#!/bin/bash

# AIDA Model Experiments - Single Encoder Across All Datasets
# Usage: ./run_aida_single_encoder.sh <encoder_name>

set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <encoder_name>"
    echo "Available encoders: mlp, tabm, dfm, resnet, fttrans, armnet"
    echo "Example: $0 mlp"
    exit 1
fi

ENCODER=$1

# Validate encoder
if [[ ! "$ENCODER" =~ ^(mlp|tabm|dfm|resnet|fttrans|armnet)$ ]]; then
    echo "Error: Invalid encoder '$ENCODER'"
    echo "Available encoders: mlp, tabm, dfm, resnet, fttrans, armnet"
    exit 1
fi

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
COMMON_ARGS="--no_need_test --verbose"

echo "=========================================="
echo "AIDA Experiments with Encoder: $ENCODER"
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

    echo "Dataset: $db_name | Task: $task_name (Classification)"
    echo "------------------------------------------"

    python -m aida.aida_run \
        --db_name "$db_name" \
        --tf_cache_dir "$tf_cache_dir" \
        --task_name "$task_name" \
        --base_encoder "$ENCODER" \
        $COMMON_ARGS

    echo "✓ Completed: $db_name - $task_name"
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

    echo "Dataset: $db_name | Task: $task_name (Regression)"
    echo "------------------------------------------"

    python -m aida.aida_run \
        --db_name "$db_name" \
        --tf_cache_dir "$tf_cache_dir" \
        --task_name "$task_name" \
        --base_encoder "$ENCODER" \
        $COMMON_ARGS

    echo "✓ Completed: $db_name - $task_name"
    echo ""
done

echo "=========================================="
echo "All datasets tested with $ENCODER encoder"
echo "=========================================="
