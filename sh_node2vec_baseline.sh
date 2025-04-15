#!/bin/bash
export PYTHONPATH=$(pwd)

# # event Database
EVENT_TASK_NAMES=("user-repeat" "user-ignore" "user-attendance")
DBNAME="event"
for TASK_NAME in "${EVENT_TASK_NAMES[@]}"; do
    echo "--------------Running task: $TASK_NAME in Database $DBNAME ------------------"
    python ./cmd/node2vec_baseline.py \
    --tf_cache_dir ./data/rel-event-tensor-frame \
    --db_name event \
    --task_name $TASK_NAME 
    echo "-------------Finished task: $TASK_NAME------------------"
    echo
done



# trail database
TRIAL_TASK_NAMES=("study-outcome" "study-adverse" "site-success")
DBNAME="trial"
for TASK_NAME in "${TRIAL_TASK_NAMES[@]}"; do
    echo "--------------Running task: $TASK_NAME in Database $DBNAME ------------------"
    python ./cmd/node2vec_baseline.py \
    --tf_cache_dir ./data/rel-trial-tensor-frame \
    --db_name trial \
    --task_name $TASK_NAME 
    echo  "-------------Finished task: $TASK_NAME------------------"
    echo
done


# Avito database
AVITO_TASK_NAMES=("user-clicks" "user-visits" "ad-ctr")
DBNAME="avito"
for TASK_NAME in "${AVITO_TASK_NAMES[@]}"; do
    echo "--------------Running task: $TASK_NAME in Database $DBNAME ------------------"
    python ./cmd/node2vec_baseline.py \
    --tf_cache_dir ./data/rel-avito-tensor-frame \
    --db_name avito \
    --channel 25 \
    --task_name $TASK_NAME \
    --batch_size 128 
    echo  "-------------Finished task: $TASK_NAME------------------"
    echo
done


# Stack
STACK_TASK_NAMES=("user-badge" "user-engagement" "post-votes")
DBNAME="stack"
for TASK_NAME in "${STACK_TASK_NAMES[@]}"; do
    echo "--------------Running task: $TASK_NAME in Database $DBNAME ------------------"
    python ./cmd/node2vec_baseline.py \
    --tf_cache_dir ./data/stack-tensor-frame \
    --data_cache_dir /home/lingze/.cache/relbench/stack \
    --db_name $DBNAME \
    --task_name $TASK_NAME \
    --channel 128 \
    --batch_size 128  
    echo  "-------------Finished task: $TASK_NAME------------------"
    echo
done