#!/bin/bash

# TP-BERTa Medium Baseline Training Script
# Usage: ./tpberta_medium_baseline.sh [dataset_name]
#   If dataset_name is provided, train only that dataset
#   Otherwise, train all datasets

set -e  # Exit on error

# ============================================
# Configuration
# ============================================

# Data directory roots (full paths)
INPUT_DATA_DIR_ROOT="/home/naili/sharing-embedding-table/data/tpberta_table"
ORIGINAL_DATA_DIR_ROOT="/home/lingze/embedding_fusion/data/fit-medium-table"

# List of datasets to train
DATA_LIST=(
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

# Check if a specific dataset is provided
SPECIFIC_DATASET="${1:-}"

# TP-BERTa paths (hard coded, server path)
TPBERTA_ROOT="/home/naili/tp-berta"
export TPBERTA_ROOT="$TPBERTA_ROOT"
export TPBERTA_PRETRAIN_DIR="$TPBERTA_ROOT/checkpoints/tp-joint"
export TPBERTA_BASE_MODEL_DIR="$TPBERTA_ROOT/checkpoints/roberta-base"
export PYTHONPATH="$TPBERTA_ROOT:$PYTHONPATH"

# Output directory for training results
RESULT_DIR="/home/naili/sharing-embedding-table/result_raw_from_server"

# Logging setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="/home/naili/sharing-embedding-table/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Determine datasets to process
if [ -n "$SPECIFIC_DATASET" ]; then
    # Train only the specified dataset
    DATASETS_TO_PROCESS=("$SPECIFIC_DATASET")
    LOG_FILE="$LOG_DIR/tpberta_medium_baseline_${SPECIFIC_DATASET}_${TIMESTAMP}.log"
    echo "=========================================="
    echo "TP-BERTa Medium Baseline Training - Single Dataset"
    echo "Dataset: $SPECIFIC_DATASET"
else
    # Train all datasets
    DATASETS_TO_PROCESS=("${DATA_LIST[@]}")
    LOG_FILE="$LOG_DIR/tpberta_medium_baseline_all_${TIMESTAMP}.log"
    echo "=========================================="
    echo "TP-BERTa Medium Baseline Training - All Datasets"
fi

# Redirect all output to log file AND console
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to: $LOG_FILE"
echo "=========================================="
echo ""

# Set CUDA_VISIBLE_DEVICES to use only one GPU
export CUDA_VISIBLE_DEVICES=0

# ============================================
# Function to train on a single dataset
# ============================================

train_dataset() {
    local dataset=$1
    local input_dir="${INPUT_DATA_DIR_ROOT}/${dataset}"
    local original_data_dir="${ORIGINAL_DATA_DIR_ROOT}/${dataset}"
    local output_dir="${RESULT_DIR}/tpberta_head/${dataset}"
    local target_col_txt="${original_data_dir}/target_col.txt"
    
    echo ""
    echo "=========================================="
    echo "Training Dataset: $dataset"
    echo "=========================================="
    echo "  INPUT_DIR: $input_dir"
    echo "  OUTPUT_DIR: $output_dir"
    echo "  TARGET_COL_TXT: $target_col_txt"
    echo ""

    # Check input directory exists
    if [ ! -d "$input_dir" ]; then
        echo "  ⚠️  Warning: Input directory not found: $input_dir"
        echo "  Skipping..."
        return
    fi
    
    # Check required files exist
    if [ ! -f "$input_dir/train.csv" ] || [ ! -f "$input_dir/val.csv" ] || [ ! -f "$input_dir/test.csv" ]; then
        echo "  ⚠️  Warning: Missing CSV files in: $input_dir"
        echo "  Skipping..."
        return
    fi
    
    if [ ! -f "$target_col_txt" ]; then
        echo "  ⚠️  Warning: target_col.txt not found: $target_col_txt"
        echo "  Skipping..."
        return
    fi
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Run training
    python "$PROJECT_ROOT/tpberta/train.py" \
        --data_dir "$input_dir" \
        --output_dir "$output_dir" \
        --target_col_txt "$target_col_txt"
    
    echo ""
    echo "  ✅ Completed: $dataset"
    echo "     Results saved to: $output_dir"
}

# ============================================
# Main - Loop through datasets to process
# ============================================

for dataset in "${DATASETS_TO_PROCESS[@]}"; do
    train_dataset "$dataset"
done

echo ""
echo "=========================================="
echo "All Datasets Training Completed!"
echo "=========================================="
echo "Results saved to: ${RESULT_DIR}/tpberta_head/"
echo "Log saved to: $LOG_FILE"
echo "=========================================="

# ============================================
# Main
# ============================================

echo "=========================================="
echo "TP-BERTa Medium Baseline Training"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Dataset: $TEST_DATASET"
echo "  INPUT_DIR: $INPUT_DIR"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  TARGET_COL_TXT: $TARGET_COL_TXT"
echo "  (Using default training parameters from train.py)"
echo ""

# Check input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ Error: Input directory not found: $INPUT_DIR"
    exit 1
fi

# Check required files exist
if [ ! -f "$INPUT_DIR/train.csv" ] || [ ! -f "$INPUT_DIR/val.csv" ] || [ ! -f "$INPUT_DIR/test.csv" ]; then
    echo "❌ Error: Missing CSV files in: $INPUT_DIR"
    echo "   Required: train.csv, val.csv, test.csv"
    exit 1
fi

if [ ! -f "$TARGET_COL_TXT" ]; then
    echo "❌ Error: target_col.txt not found: $TARGET_COL_TXT"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Set CUDA_VISIBLE_DEVICES to use only one GPU
export CUDA_VISIBLE_DEVICES=0

# Run training
echo "=========================================="
echo "Running TP-BERTa Medium Baseline Training"
echo "=========================================="
echo ""
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

python "$PROJECT_ROOT/tpberta/train.py" \
    --data_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --target_col_txt "$TARGET_COL_TXT"

echo ""
echo "=========================================="
echo "Training completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "  - results.json (training metrics)"
echo "  - test_predictions.npy"
echo "  - test_targets.npy"
echo "Log saved to: $LOG_FILE"
echo "=========================================="

