#!/bin/bash

# TP-BERTa Preprocessing Test Script
# Usage: ./generate_medium_tpbert_table.sh [dataset_name]

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
TEST_DATASET="${1:-avito-ad-ctr}"
LOG_FILE="$LOG_DIR/generate_tpbert_table_${TEST_DATASET}_${TIMESTAMP}.log"

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

# Data source directories (hard coded like exam.sh)
DATA_DIRS=(
  "fit-medium-table" 
)

# Default data source (first one)
DATA_SOURCE="${DATA_DIRS[0]}"

# Output directory for TP-BERTa format files (hard coded)
OUTPUT_BASE_DIR="/home/naili/sharing-embedding-table/data/tpberta_table"
OUTPUT_DIR="$OUTPUT_BASE_DIR/$TEST_DATASET"

# Test dataset (already set above for logging)
# TEST_DATASET is set at line 20 from command line argument

# ============================================
# Main
# ============================================

echo "=========================================="
echo "TP-BERTa Preprocessing - Convert to TP-BERTa Format"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Dataset: $TEST_DATASET"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo ""

# Try to find dataset in any data source (hard coded like exam.sh)
INPUT_DIR=""
for DATA_SOURCE in "${DATA_DIRS[@]}"; do
    DATA_PATH="/home/lingze/embedding_fusion/data/${DATA_SOURCE}/${TEST_DATASET}"
    if [ -d "$DATA_PATH" ]; then
        INPUT_DIR="$DATA_PATH"
        echo "  Found dataset in: $DATA_SOURCE"
        echo "  INPUT_DIR: $INPUT_DIR"
        break
    fi
done

# Check dataset exists
if [ -z "$INPUT_DIR" ]; then
    echo "❌ Error: Dataset not found in any data source: $TEST_DATASET"
    echo "   Searched in: ${DATA_DIRS[@]}"
    exit 1
fi

# Check required files exist
if [ ! -f "$INPUT_DIR/train.csv" ] || [ ! -f "$INPUT_DIR/val.csv" ] || [ ! -f "$INPUT_DIR/test.csv" ]; then
    echo "❌ Error: Missing CSV files in: $INPUT_DIR"
    echo "   Required: train.csv, val.csv, test.csv"
    exit 1
fi

if [ ! -f "$INPUT_DIR/target_col.txt" ]; then
    echo "❌ Error: Missing target_col.txt in: $INPUT_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run preprocessing
echo "=========================================="
echo "Running TP-BERTa Preprocessing"
echo "=========================================="
echo ""
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Set CUDA_VISIBLE_DEVICES to use only one GPU (avoid DataParallel)
export CUDA_VISIBLE_DEVICES=0

python "$PROJECT_ROOT/tpberta/preprocess.py" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --pretrain_dir "$TPBERTA_PRETRAIN_DIR"

echo ""
echo "=========================================="
echo "Preprocessing completed!"
echo "Output saved to: $OUTPUT_DIR"
echo "  - train.csv (2 columns: embedding, target)"
echo "  - val.csv (2 columns: embedding, target)"
echo "  - test.csv (2 columns: embedding, target)"
echo "  - feature_names.json"
echo "Log saved to: $LOG_FILE"
echo "=========================================="

