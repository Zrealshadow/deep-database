#!/bin/bash

# Retry Failed Tasks Script
# Processes only the tasks that failed in the initial run
# Usage: ./retry_failed_tasks.sh

set -e  # Exit on error (but we'll handle errors in the loop)

# ============================================
# Configuration
# ============================================

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# TP-BERTa paths
TPBERTA_ROOT="/home/naili/tp-berta"
export TPBERTA_ROOT="$TPBERTA_ROOT"
export TPBERTA_PRETRAIN_DIR="$TPBERTA_ROOT/checkpoints/tp-joint"
export TPBERTA_BASE_MODEL_DIR="$TPBERTA_ROOT/checkpoints/roberta-base"
export PYTHONPATH="$PROJECT_ROOT:$TPBERTA_ROOT:$PYTHONPATH"

# Set CUDA_VISIBLE_DEVICES to use only one GPU
export CUDA_VISIBLE_DEVICES=0

# Logging setup
LOG_DIR="/home/naili/sharing-embedding-table/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/retry_failed_tasks_${TIMESTAMP}.log"

# Redirect all output to log file AND console
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=========================================="
echo "Retry Failed Tasks"
echo "=========================================="
echo "Logging to: $LOG_FILE"
echo "=========================================="
echo ""

# ============================================
# Failed Tasks
# ============================================
# trial:study-outcome - Failed on: nomic, tpberta (BGE succeeded)

echo "=========================================="
echo "Task 1: trial:study-outcome (Nomic) - batch_size=1"
echo "=========================================="
python "$PROJECT_ROOT/scripts/retry_failed_task.py" \
    --db_name trial \
    --task_name study-outcome \
    --cache_dir rel-trial \
    --model nomic \
    --batch_size 1
echo ""

echo "=========================================="
echo "Task 2: trial:study-outcome (TP-BERTa)"
echo "=========================================="
python "$PROJECT_ROOT/scripts/retry_failed_task.py" \
    --db_name trial \
    --task_name study-outcome \
    --cache_dir rel-trial \
    --model tpberta
echo ""

echo "=========================================="
echo "All Failed Tasks Processing Completed!"
echo "=========================================="
echo "Log saved to: $LOG_FILE"
echo "=========================================="
