#!/bin/bash

# AIDA Model Experiments - Single Dataset with All Encoders
# Usage: ./run_aida_single_dataset.sh <db_name> <tf_cache_dir> <task_name>

set -e

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <db_name> <tf_cache_dir> <task_name>"
    echo "Example: $0 hm data/hm-tensor-frame user-churn"
    exit 1
fi

DB_NAME=$1
TF_CACHE_DIR=$2
TASK_NAME=$3

# Define base encoders to test
ENCODERS=("mlp" "tabm" "dfm" "resnet" "fttrans")

# Optional: Add common arguments here
COMMON_ARGS="--no_need_test --verbose"

echo "=========================================="
echo "AIDA Experiments: $DB_NAME - $TASK_NAME"
echo "=========================================="
echo ""

# Loop through all encoders
for encoder in "${ENCODERS[@]}"; do
    echo "Running with encoder: $encoder"

    python -m aida.aida_run \
        --db_name "$DB_NAME" \
        --tf_cache_dir "$TF_CACHE_DIR" \
        --task_name "$TASK_NAME" \
        --base_encoder "$encoder" \
        $COMMON_ARGS

    echo "✓ Completed: $encoder"
    echo ""
done

echo "=========================================="
echo "All encoders tested for $DB_NAME - $TASK_NAME"
echo "=========================================="
