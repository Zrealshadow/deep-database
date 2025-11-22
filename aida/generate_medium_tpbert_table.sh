#!/bin/bash

# TP-BERTa Preprocessing Script
# Usage: ./generate_medium_tpbert_table.sh [dataset_name]
#   If dataset_name is provided, process only that dataset
#   Otherwise, process all datasets

set -e  # Exit on error

# ============================================
# Configuration
# ============================================

# Data directory root (full path)
DATA_DIR_ROOT="/home/lingze/embedding_fusion/data/fit-medium-table"

# List of datasets to process
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

# Output directory for TP-BERTa format files (hard coded)
OUTPUT_BASE_DIR="/home/naili/sharing-embedding-table/data/tpberta_table"

# Logging setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="/home/naili/sharing-embedding-table/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Determine datasets to process
if [ -n "$SPECIFIC_DATASET" ]; then
    # Process only the specified dataset
    DATASETS_TO_PROCESS=("$SPECIFIC_DATASET")
    LOG_FILE="$LOG_DIR/generate_tpbert_table_${SPECIFIC_DATASET}_${TIMESTAMP}.log"
    echo "=========================================="
    echo "TP-BERTa Preprocessing - Single Dataset"
    echo "Dataset: $SPECIFIC_DATASET"
else
    # Process all datasets
    DATASETS_TO_PROCESS=("${DATA_LIST[@]}")
    LOG_FILE="$LOG_DIR/generate_tpbert_table_all_${TIMESTAMP}.log"
    echo "=========================================="
    echo "TP-BERTa Preprocessing - All Datasets"
fi

# Redirect all output to log file AND console
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to: $LOG_FILE"
echo "=========================================="
echo ""

# Set CUDA_VISIBLE_DEVICES to use only one GPU (avoid DataParallel)
export CUDA_VISIBLE_DEVICES=0

# ============================================
# Function to process a single dataset
# ============================================

process_dataset() {
    local dataset=$1
    local input_dir="${DATA_DIR_ROOT}/${dataset}"
    local output_dir="${OUTPUT_BASE_DIR}/${dataset}"
    
    echo ""
    echo "=========================================="
    echo "Processing Dataset: $dataset"
    echo "=========================================="
    echo "  INPUT_DIR: $input_dir"
    echo "  OUTPUT_DIR: $output_dir"
    echo ""
    
    # Check dataset exists
    if [ ! -d "$input_dir" ]; then
        echo "  ⚠️  Warning: Dataset directory not found: $input_dir"
        echo "  Skipping..."
        return
    fi
    
    # Check required files exist
    if [ ! -f "$input_dir/train.csv" ] || [ ! -f "$input_dir/val.csv" ] || [ ! -f "$input_dir/test.csv" ]; then
        echo "  ⚠️  Warning: Missing CSV files in: $input_dir"
        echo "  Skipping..."
        return
    fi
    
    if [ ! -f "$input_dir/target_col.txt" ]; then
        echo "  ⚠️  Warning: Missing target_col.txt in: $input_dir"
        echo "  Skipping..."
        return
    fi
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Run preprocessing
    python "$PROJECT_ROOT/tpberta/preprocess.py" \
        --input_dir "$input_dir" \
        --output_dir "$output_dir"
    
    echo ""
    echo "  ✅ Completed: $dataset"
    echo "     Output saved to: $output_dir"
}

# ============================================
# Main - Loop through datasets to process
# ============================================

for dataset in "${DATASETS_TO_PROCESS[@]}"; do
    process_dataset "$dataset"
done

echo ""
echo "=========================================="
echo "All Datasets Processing Completed!"
echo "=========================================="
echo "Results saved to: $OUTPUT_BASE_DIR"
echo "Log saved to: $LOG_FILE"
echo "=========================================="

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
    --output_dir "$OUTPUT_DIR"

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

