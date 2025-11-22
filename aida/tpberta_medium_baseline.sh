#!/bin/bash

# TP-BERTa Medium Baseline Training Script
# Usage: ./tpberta_medium_baseline.sh [dataset_name]

set -e  # Exit on error

# ============================================
# Logging Setup
# ============================================

# Create logs directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="/home/naili/sharing-embedding-table/logs"
mkdir -p "$LOG_DIR"

# Generate log file name with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TEST_DATASET="${1:-event-user-repeat}"
LOG_FILE="$LOG_DIR/tpberta_medium_baseline_${TEST_DATASET}_${TIMESTAMP}.log"

# Redirect all output to log file AND console
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=========================================="
echo "Logging to: $LOG_FILE"
echo "=========================================="
echo ""

# ============================================
# Configuration
# ============================================

# TP-BERTa paths (hard coded, server path)
TPBERTA_ROOT="/home/naili/tp-berta"
export TPBERTA_ROOT="$TPBERTA_ROOT"
export TPBERTA_PRETRAIN_DIR="$TPBERTA_ROOT/checkpoints/tp-joint"
export TPBERTA_BASE_MODEL_DIR="$TPBERTA_ROOT/checkpoints/roberta-base"
export PYTHONPATH="$TPBERTA_ROOT:$PYTHONPATH"

# Input directory (where preprocessed embeddings are stored)
INPUT_BASE_DIR="/home/naili/sharing-embedding-table/data/tpberta_table"
INPUT_DIR="$INPUT_BASE_DIR/$TEST_DATASET"

# Data source directories (hard coded like generate_medium_tpbert_table.sh)
DATA_DIRS=(
  "fit-medium-table"
)

# Find original data directory to get target_col.txt
ORIGINAL_DATA_DIR=""
for DATA_SOURCE in "${DATA_DIRS[@]}"; do
    DATA_PATH="/home/lingze/embedding_fusion/data/${DATA_SOURCE}/${TEST_DATASET}"
    if [ -d "$DATA_PATH" ]; then
        ORIGINAL_DATA_DIR="$DATA_PATH"
        break
    fi
done

if [ -z "$ORIGINAL_DATA_DIR" ]; then
    echo "❌ Error: Original data directory not found for dataset: $TEST_DATASET"
    echo "   Searched in: ${DATA_DIRS[@]}"
    exit 1
fi

TARGET_COL_TXT="$ORIGINAL_DATA_DIR/target_col.txt"

# Output directory for training results
RESULT_DIR="/home/naili/sharing-embedding-table/result_raw_from_server"
OUTPUT_DIR="$RESULT_DIR/tpberta_head/$TEST_DATASET"

# Test dataset (already set above for logging)
# TEST_DATASET is set at line 20 from command line argument

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

# Run training
echo "=========================================="
echo "Running TP-BERTa Medium Baseline Training"
echo "=========================================="
echo ""
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Set CUDA_VISIBLE_DEVICES to use only one GPU
export CUDA_VISIBLE_DEVICES=0

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

