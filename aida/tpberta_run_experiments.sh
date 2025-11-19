#!/bin/bash

# TP-BERTa Experiments on TableData
# This script runs TP-BERTa experiments across multiple datasets

set -e  # Exit on error

# ============================================
# Configuration
# ============================================

# Set TP-BERTa paths
# Server path: /home/naili/tp-berta
TPBERTA_ROOT="${TPBERTA_ROOT:-/home/naili/tp-berta}"

# Export environment variables
export TPBERTA_ROOT="$TPBERTA_ROOT"
export TPBERTA_PRETRAIN_DIR="${TPBERTA_PRETRAIN_DIR:-$TPBERTA_ROOT/checkpoints/tp-joint}"
export TPBERTA_BASE_MODEL_DIR="${TPBERTA_BASE_MODEL_DIR:-$TPBERTA_ROOT/checkpoints/roberta-base}"
export PYTHONPATH="$TPBERTA_ROOT:$PYTHONPATH"

# Project root (for data and results)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Data directory root (adjust if your data is in a different location)
DATA_DIR_ROOT="${DATA_DIR_ROOT:-$PROJECT_ROOT/data/dfs-flatten-table}"

# Result directory
RESULT_DIR="${RESULT_DIR:-$PROJECT_ROOT/tpberta_outputs}"

# Training parameters
MAX_EPOCHS="${MAX_EPOCHS:-200}"
EARLY_STOP="${EARLY_STOP:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"

# ============================================
# Dataset Configuration
# ============================================

# Classification datasets (from TableData format)
CLASSIFICATION_DATASETS=(
    "avito-user-clicks"
    "avito-ad-ctr"
    "event-user-repeat"
    "event-user-ignore"
    "ratebeer-place-positive"
    "ratebeer-user-active"
    "trial-study-outcome"
    "trial-study-adverse"
    "hm-user-churn"
)

# Regression datasets (from TableData format)
REGRESSION_DATASETS=(
    "avito-ad-ctr"
    "event-user-attendance"
    "ratebeer-beer-positive"
    "trial-site-success"
    "f1-driver-dnf"
    "f1-driver-top3"
    "hm-item-sales"
    "amazon-user-churn"
    "amazon-item-churn"
)

# ============================================
# Helper Functions
# ============================================

# Function to check if dataset exists
check_dataset() {
    local dataset=$1
    local data_dir="${DATA_DIR_ROOT}/${dataset}"
    
    if [ ! -d "$data_dir" ]; then
        echo "⚠️  Dataset not found: $data_dir"
        return 1
    fi
    
    if [ ! -f "$data_dir/train.csv" ] || [ ! -f "$data_dir/val.csv" ] || [ ! -f "$data_dir/test.csv" ]; then
        echo "⚠️  Missing CSV files in: $data_dir"
        return 1
    fi
    
    if [ ! -f "$data_dir/target_col.txt" ]; then
        echo "⚠️  Missing target_col.txt in: $data_dir"
        return 1
    fi
    
    return 0
}

# Function to run TP-BERTa experiment
run_tpberta() {
    local dataset=$1
    local data_dir="${DATA_DIR_ROOT}/${dataset}"
    
    echo "  Running TP-BERTa on: $dataset"
    
    python "$PROJECT_ROOT/cmds/tpberta_train.py" \
        --data_dir "$data_dir" \
        --result_dir "$RESULT_DIR" \
        --max_epochs "$MAX_EPOCHS" \
        --early_stop "$EARLY_STOP" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LEARNING_RATE"
}

# ============================================
# Main Execution
# ============================================

echo "=========================================="
echo "TP-BERTa Experiments"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  TPBERTA_ROOT: $TPBERTA_ROOT"
echo "  TPBERTA_PRETRAIN_DIR: $TPBERTA_PRETRAIN_DIR"
echo "  DATA_DIR_ROOT: $DATA_DIR_ROOT"
echo "  RESULT_DIR: $RESULT_DIR"
echo "  MAX_EPOCHS: $MAX_EPOCHS"
echo "  EARLY_STOP: $EARLY_STOP"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "  LEARNING_RATE: $LEARNING_RATE"
echo ""

# Check TP-BERTa installation
if [ ! -d "$TPBERTA_ROOT" ]; then
    echo "❌ Error: TP-BERTa root not found: $TPBERTA_ROOT"
    echo "   Please ensure tp-berta is installed at: $TPBERTA_ROOT"
    echo "   Or set TPBERTA_ROOT environment variable"
    exit 1
fi

if [ ! -f "$TPBERTA_ROOT/bin/tpberta_modeling.py" ]; then
    echo "❌ Error: TP-BERTa source code not found in: $TPBERTA_ROOT"
    echo "   Expected: $TPBERTA_ROOT/bin/tpberta_modeling.py"
    exit 1
fi

# Check pre-trained model
if [ ! -f "$TPBERTA_PRETRAIN_DIR/pytorch_models/best/pytorch_model.bin" ]; then
    echo "⚠️  Warning: Pre-trained model not found: $TPBERTA_PRETRAIN_DIR/pytorch_models/best/pytorch_model.bin"
    echo "   Run: cd $PROJECT_ROOT/aida && ./tpberta_download.sh"
    echo ""
fi

# Process Classification Datasets
echo "=========================================="
echo "Classification Tasks"
echo "=========================================="
echo ""

for dataset in "${CLASSIFICATION_DATASETS[@]}"; do
    if check_dataset "$dataset"; then
        run_tpberta "$dataset"
        echo ""
    else
        echo "  ⏭️  Skipping: $dataset"
        echo ""
    fi
done

echo ""
echo "=========================================="
echo "Regression Tasks"
echo "=========================================="
echo ""

# Process Regression Datasets
for dataset in "${REGRESSION_DATASETS[@]}"; do
    if check_dataset "$dataset"; then
        run_tpberta "$dataset"
        echo ""
    else
        echo "  ⏭️  Skipping: $dataset"
        echo ""
    fi
done

echo "=========================================="
echo "All TP-BERTa experiments completed!"
echo "=========================================="
echo ""
echo "Results saved to: $RESULT_DIR"

