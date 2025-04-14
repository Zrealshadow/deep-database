#!/bin/bash

MODEL="GAT"

# event Database
EVENT_TASK_NAMES=("user-repeat" "user-ignore" "user-attendance")

for TASK_NAME in "${EVENT_TASK_NAMES[@]}"; do
    echo "--------------Running task: $TASK_NAME------------------"
    python ./cmd/graph_baseline.py \
    --tf_cache_dir ./data/rel-event-tensor-frame \
    --db_name event \
    --task_name $TASK_NAME \
    --model $MODEL
    echo  "-------------Finished task: $TASK_NAME------------------"
    echo
done




# trail database
TRIAL_TASK_NAMES=("study-outcome" "study-adverse" "site-success")

for TASK_NAME in "${TRIAL_TASK_NAMES[@]}"; do
    echo "--------------Running task: $TASK_NAME------------------"
    python ./cmd/graph_baseline.py \
    --tf_cache_dir ./data/rel-trial-tensor-frame \
    --db_name trial \
    --task_name $TASK_NAME \
    --model $MODEL
    echo  "-------------Finished task: $TASK_NAME------------------"
    echo
done




# Avito database
AVITO_TASK_NAMES=("user-clicks" "user-visits" "ad-ctr")

for TASK_NAME in "${AVITO_TASK_NAMES[@]}"; do
    echo "--------------Running task: $TASK_NAME------------------"
    python ./cmd/graph_baseline.py \
    --tf_cache_dir ./data/rel-avito-tensor-frame \
    --db_name avito \
    --task_name $TASK_NAME \
    --model $MODEL
    echo  "-------------Finished task: $TASK_NAME------------------"
    echo
done



# Stack
STACK_TASK_NAMES=("user-badge" "user-engagement" "post-votes")

for TASK_NAME in "${STACK_TASK_NAMES[@]}"; do
    echo "--------------Running task: $TASK_NAME------------------"
    python ./cmd/graph_baseline.py \
    --tf_cache_dir ./data/stack-tensor-frame \
    --data_cache_dir /home/lingze/.cache/relbench/stack \
    --db_name stack \
    --task_name $TASK_NAME \
    --model $MODEL
    echo  "-------------Finished task: $TASK_NAME------------------"
    echo
done