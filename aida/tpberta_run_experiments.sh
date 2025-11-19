#!/bin/bash

# TP-BERTa Experiments on TableData
# This script runs TP-BERTa experiments across multiple datasets

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
LOG_FILE="$LOG_DIR/tpberta_experiments_${TIMESTAMP}.log"

# Redirect all output to log file AND console
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=========================================="
echo "Logging to: $LOG_FILE"
echo "=========================================="
echo ""

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

# Project root (already set above for logging)

# Data source directories (hard coded like exam.sh)
DATA_DIRS=(
  "fit-best-table"
  "fit-medium-table" 
  "flatten-table"
)

# Result directory (save all results here)
RESULT_DIR="${RESULT_DIR:-/home/naili/sharing-embedding-table/result_raw_from_server}"

# Training parameters
MAX_EPOCHS="${MAX_EPOCHS:-200}"
EARLY_STOP="${EARLY_STOP:-50}"
BATCH_SIZE="${BATCH_SIZE:-4}"      # Reduced for large feature sets (some datasets have 500+ features)
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

# Function to check if dataset exists (hard coded like exam.sh)
check_dataset() {
    local dataset=$1
    local data_source=$2
    local data_path="/home/lingze/embedding_fusion/data/${data_source}/${dataset}"
    
    if [ ! -d "$data_path" ]; then
        return 1
    fi
    
    if [ ! -f "$data_path/train.csv" ] || [ ! -f "$data_path/val.csv" ] || [ ! -f "$data_path/test.csv" ]; then
        return 1
    fi
    
    if [ ! -f "$data_path/target_col.txt" ]; then
        return 1
    fi
    
    return 0
}

# Function to run TP-BERTa experiment (hard coded like exam.sh)
run_tpberta() {
    local dataset=$1
    local data_source=$2
    local data_path="/home/lingze/embedding_fusion/data/${data_source}/${dataset}"
    
    echo "  Running TP-BERTa on: $dataset (${data_source})"
    
    python "$PROJECT_ROOT/cmds/tpberta_train.py" \
        --data_dir "$data_path" \
        --result_dir "$RESULT_DIR" \
        --max_epochs "$MAX_EPOCHS" \
        --early_stop "$EARLY_STOP" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LEARNING_RATE" \
        --freeze_encoder
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
echo "  Data Sources: ${DATA_DIRS[@]}"
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

# Process all datasets from all data sources (hard coded like exam.sh)
for DATA_SOURCE in "${DATA_DIRS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Processing ${DATA_SOURCE} datasets..."
    echo "=========================================="
    echo ""
    
    # Process Classification Datasets
    echo "Classification Tasks (${DATA_SOURCE}):"
    echo ""
    
    for dataset in "${CLASSIFICATION_DATASETS[@]}"; do
        if check_dataset "$dataset" "$DATA_SOURCE"; then
            run_tpberta "$dataset" "$DATA_SOURCE"
            echo ""
        else
            echo "  ⏭️  Skipping ${dataset} (not found in ${DATA_SOURCE})"
            echo ""
        fi
    done
    
    echo ""
    echo "Regression Tasks (${DATA_SOURCE}):"
    echo ""
    
    # Process Regression Datasets
    for dataset in "${REGRESSION_DATASETS[@]}"; do
        if check_dataset "$dataset" "$DATA_SOURCE"; then
            run_tpberta "$dataset" "$DATA_SOURCE"
            echo ""
        else
            echo "  ⏭️  Skipping ${dataset} (not found in ${DATA_SOURCE})"
            echo ""
        fi
    done
done

echo "=========================================="
echo "All TP-BERTa experiments completed!"
echo "=========================================="
echo ""
echo "Results saved to: $RESULT_DIR"
echo "Log saved to: $LOG_FILE"

