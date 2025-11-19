#!/bin/bash

# TP-BERTa Test Script - Run on a single dataset
# Usage: ./tpberta_test.sh [dataset_name]

set -e  # Exit on error

# ============================================
# Logging Setup
# ============================================

# Create logs directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"
mkdir -p "$LOG_DIR"

# Generate log file name with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TEST_DATASET="${1:-avito-ad-ctr}"
LOG_FILE="$LOG_DIR/tpberta_test_${TEST_DATASET}_${TIMESTAMP}.log"

# Redirect all output to log file AND console
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=========================================="
echo "Logging to: $LOG_FILE"
echo "=========================================="
echo ""

# ============================================
# Configuration
# ============================================

# TP-BERTa paths (server path)
TPBERTA_ROOT="${TPBERTA_ROOT:-/home/naili/tp-berta}"
export TPBERTA_ROOT="$TPBERTA_ROOT"
export TPBERTA_PRETRAIN_DIR="${TPBERTA_PRETRAIN_DIR:-$TPBERTA_ROOT/checkpoints/tp-joint}"
export TPBERTA_BASE_MODEL_DIR="${TPBERTA_BASE_MODEL_DIR:-$TPBERTA_ROOT/checkpoints/roberta-base}"
export PYTHONPATH="$TPBERTA_ROOT:$PYTHONPATH"

# Project root (already set above for logging)

# Data source directories (hard coded like exam.sh)
DATA_DIRS=(
  "fit-best-table"
  "fit-medium-table" 
  "flatten-table"
)

# Default data source (first one)
DATA_SOURCE="${DATA_DIRS[0]}"

# Result directory
RESULT_DIR="${RESULT_DIR:-$PROJECT_ROOT/tpberta_outputs}"

# Training parameters (reduced for testing)
MAX_EPOCHS="${MAX_EPOCHS:-5}"      # Reduced for quick test
EARLY_STOP="${EARLY_STOP:-3}"      # Reduced for quick test
BATCH_SIZE="${BATCH_SIZE:-4}"      # Reduced for large feature sets (536 features need smaller batch)
LEARNING_RATE="${LEARNING_RATE:-1e-5}"

# Test dataset (already set above for logging)
# TEST_DATASET is set at line 20 from command line argument

# ============================================
# Main
# ============================================

echo "=========================================="
echo "TP-BERTa Test - Single Dataset"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Dataset: $TEST_DATASET"
echo "  TPBERTA_ROOT: $TPBERTA_ROOT"
echo "  RESULT_DIR: $RESULT_DIR"
echo "  MAX_EPOCHS: $MAX_EPOCHS (reduced for testing)"
echo "  EARLY_STOP: $EARLY_STOP (reduced for testing)"
echo ""

# Try to find dataset in any data source (hard coded like exam.sh)
DATA_DIR=""
for DATA_SOURCE in "${DATA_DIRS[@]}"; do
    DATA_PATH="/home/lingze/embedding_fusion/data/${DATA_SOURCE}/${TEST_DATASET}"
    if [ -d "$DATA_PATH" ]; then
        DATA_DIR="$DATA_PATH"
        echo "  Found dataset in: $DATA_SOURCE"
        echo "  DATA_DIR: $DATA_DIR"
        break
    fi
done

# Check dataset exists
if [ -z "$DATA_DIR" ]; then
    echo "❌ Error: Dataset not found in any data source: $TEST_DATASET"
    echo "   Searched in: ${DATA_DIRS[@]}"
    exit 1
fi
if [ ! -f "$DATA_DIR/train.csv" ] || [ ! -f "$DATA_DIR/val.csv" ] || [ ! -f "$DATA_DIR/test.csv" ]; then
    echo "❌ Error: Missing CSV files in: $DATA_DIR"
    exit 1
fi

# Check TP-BERTa
if [ ! -d "$TPBERTA_ROOT" ]; then
    echo "❌ Error: TP-BERTa not found: $TPBERTA_ROOT"
    exit 1
fi

# Run test
echo "=========================================="
echo "Running TP-BERTa on: $TEST_DATASET"
echo "=========================================="
echo ""

python "$PROJECT_ROOT/cmds/tpberta_train.py" \
    --data_dir "$DATA_DIR" \
    --result_dir "$RESULT_DIR" \
    --max_epochs "$MAX_EPOCHS" \
    --early_stop "$EARLY_STOP" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE"

echo ""
echo "=========================================="
echo "Test completed!"
echo "Results saved to: $RESULT_DIR/$TEST_DATASET/results.json"
echo "Log saved to: $LOG_FILE"
echo "=========================================="

