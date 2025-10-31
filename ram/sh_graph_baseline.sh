#!/bin/bash
# Execute graph baseline experiments (GCN, GAT, HGT) across all databases and tasks
export PYTHONPATH=$(pwd)
set -e  # Exit on error

# Configuration
# Model options: GCN, GAT, HGT
MODEL="${1:-GCN}"  # Default to GCN if no argument provided

echo "========================================"
echo "Running Graph Baseline Experiments"
echo "Model: $MODEL"
echo "========================================"

# Validate model choice
if [[ ! "$MODEL" =~ ^(GCN|GAT|HGT)$ ]]; then
    echo "Error: Invalid model '$MODEL'. Must be one of: GCN, GAT, HGT"
    echo "Usage: $0 [GCN|GAT|HGT]"
    exit 1
fi

# Event Database
echo -e "\n=== Event Database ==="
EVENT_TASK_NAMES=("user-repeat" "user-ignore" "user-attendance")
DBNAME="event"
for TASK_NAME in "${EVENT_TASK_NAMES[@]}"; do
    echo "Running task: $TASK_NAME in Database $DBNAME with $MODEL"
    python -m ram.graph_baseline \
        --tf_cache_dir ./data/rel-event-tensor-frame \
        --db_name event \
        --task_name $TASK_NAME \
        --model $MODEL
    echo "Finished task: $TASK_NAME"
    echo
done

# Trial Database
echo -e "\n=== Trial Database ==="
TRIAL_TASK_NAMES=("study-outcome" "study-adverse" "site-success")
DBNAME="trial"
for TASK_NAME in "${TRIAL_TASK_NAMES[@]}"; do
    echo "Running task: $TASK_NAME in Database $DBNAME with $MODEL"
    python -m ram.graph_baseline \
        --tf_cache_dir ./data/rel-trial-tensor-frame \
        --db_name trial \
        --task_name $TASK_NAME \
        --model $MODEL
    echo "Finished task: $TASK_NAME"
    echo
done

# Avito Database
echo -e "\n=== Avito Database ==="
AVITO_TASK_NAMES=("user-clicks" "user-visits" "ad-ctr")
DBNAME="avito"
for TASK_NAME in "${AVITO_TASK_NAMES[@]}"; do
    echo "Running task: $TASK_NAME in Database $DBNAME with $MODEL"
    python -m ram.graph_baseline \
        --tf_cache_dir ./data/rel-avito-tensor-frame \
        --db_name avito \
        --task_name $TASK_NAME \
        --model $MODEL \
        --no_need_test
    echo "Finished task: $TASK_NAME"
    echo
done

# Stack Database
echo -e "\n=== Stack Database ==="
STACK_TASK_NAMES=("user-badge" "user-engagement" "post-votes")
DBNAME="stack"
for TASK_NAME in "${STACK_TASK_NAMES[@]}"; do
    echo "Running task: $TASK_NAME in Database $DBNAME with $MODEL"
    python -m ram.graph_baseline \
        --tf_cache_dir ./data/stack-tensor-frame \
        --data_cache_dir /home/lingze/.cache/relbench/stack \
        --db_name $DBNAME \
        --task_name $TASK_NAME \
        --model $MODEL \
        --no_need_test
    echo "Finished task: $TASK_NAME"
    echo
done

# RateBeer Database
echo -e "\n=== RateBeer Database ==="
RATEBEER_TASK_NAMES=("user-active" "beer-positive" "place-positive")
DBNAME="ratebeer"
for TASK_NAME in "${RATEBEER_TASK_NAMES[@]}"; do
    echo "Running task: $TASK_NAME in Database $DBNAME with $MODEL"
    python -m ram.graph_baseline \
        --tf_cache_dir ./data/ratebeer-tensor-frame \
        --data_cache_dir /home/lingze/.cache/relbench/ratebeer \
        --db_name $DBNAME \
        --task_name $TASK_NAME \
        --model $MODEL \
        --no_need_test
    echo "Finished task: $TASK_NAME"
    echo
done

echo -e "\n========================================"
echo "All graph baseline experiments completed!"
echo "Model: $MODEL"
echo "========================================"