#!/bin/bash
# Execute DNN baseline experiments on tabular data
export PYTHONPATH=$(pwd)
set -e  # Exit on error

DATA_BASE_DIR="./data/flatten-table"
MODEL="MLP"  # Default model: MLP, ResNet, FTTrans

echo "========================================"
echo "Running DNN Baseline Experiments"
echo "Model: $MODEL"
echo "Data Directory: $DATA_BASE_DIR"
echo "========================================"

# Event Dataset - Classification Tasks
echo -e "\n=== Event Dataset (Classification) ==="
python ./ram/dnn_baseline_table_data.py \
    --data_dir $DATA_BASE_DIR/event-user-ignore \
    --model $MODEL

python ./ram/dnn_baseline_table_data.py \
    --data_dir $DATA_BASE_DIR/event-user-repeat \
    --model $MODEL

# Example: Run with different models
# python ./ram/dnn_baseline_table_data.py \
#     --data_dir $DATA_BASE_DIR/event-user-repeat \
#     --model ResNet

# python ./ram/dnn_baseline_table_data.py \
#     --data_dir $DATA_BASE_DIR/event-user-repeat \
#     --model FTTrans

# Stack Dataset - Classification Tasks
echo -e "\n=== Stack Dataset (Classification) ==="
python ./ram/dnn_baseline_table_data.py \
    --data_dir $DATA_BASE_DIR/stack-user-engagement \
    --model $MODEL

python ./ram/dnn_baseline_table_data.py \
    --data_dir $DATA_BASE_DIR/stack-user-badge \
    --model $MODEL

# Trial Dataset - Classification Tasks
echo -e "\n=== Trial Dataset (Classification) ==="
python ./ram/dnn_baseline_table_data.py \
    --data_dir $DATA_BASE_DIR/trial-study-outcome \
    --model $MODEL

# Avito Dataset - Classification Tasks
echo -e "\n=== Avito Dataset (Classification) ==="
python ./ram/dnn_baseline_table_data.py \
    --data_dir $DATA_BASE_DIR/avito-user-visits \
    --model $MODEL

python ./ram/dnn_baseline_table_data.py \
    --data_dir $DATA_BASE_DIR/avito-user-clicks \
    --model $MODEL

# ===================================
# Regression Tasks
# ===================================
echo -e "\n=== Regression Tasks ==="

# Avito - Regression
echo "Running Avito regression tasks..."
python ./ram/dnn_baseline_table_data.py \
    --data_dir $DATA_BASE_DIR/avito-ad-ctr \
    --model $MODEL

# Event - Regression
echo "Running Event regression tasks..."
python ./ram/dnn_baseline_table_data.py \
    --data_dir $DATA_BASE_DIR/event-user-attendance \
    --model $MODEL

# Trial - Regression
echo "Running Trial regression tasks..."
python ./ram/dnn_baseline_table_data.py \
    --data_dir $DATA_BASE_DIR/trial-study-adverse \
    --model $MODEL

python ./ram/dnn_baseline_table_data.py \
    --data_dir $DATA_BASE_DIR/trial-site-success \
    --model $MODEL

# Stack - Regression
echo "Running Stack regression tasks..."
python ./ram/dnn_baseline_table_data.py \
    --data_dir $DATA_BASE_DIR/stack-post-votes \
    --model $MODEL

echo -e "\n========================================"
echo "All DNN baseline experiments completed!"
echo "========================================"