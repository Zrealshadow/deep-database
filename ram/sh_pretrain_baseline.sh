#!/bin/bash
# Execute graph pretraining baseline experiments (DGI, GraphCL, BGRL)
export PYTHONPATH=$(pwd)
set -e  # Exit on error

echo "========================================"
echo "Running Graph Pretraining Baselines"
echo "========================================"

METHODS=("dgi" "graphcl")

DBNAME="event"
for METHOD in "${METHODS[@]}"; do
    echo "--------------Running Pretrain Task: $METHOD in Database $DBNAME ------------------"
    python -m ram.pretrain_baseline \
    --tf_cache_dir ./data/rel-event-tensor-frame \
    --db_name $DBNAME \
    --output_dir ./static \
    --method $METHOD
    echo "-------------Finished Pretrain Task: $METHOD in Database $DBNAME------------------"
    echo
done



DBNAME="trial"
for METHOD in "${METHODS[@]}"; do
    echo "--------------Running Pretrain Task: $METHOD in Database $DBNAME ------------------"
    python -m ram.pretrain_baseline \
    --tf_cache_dir ./data/rel-trial-tensor-frame \
    --db_name $DBNAME \
    --output_dir ./static \
    --lr 0.05 \
    --max_round_epoch 20 \
    --method $METHOD
    echo "-------------Finished Pretrain Task: $METHOD in Database $DBNAME------------------"
    echo
done



DBNAME="avito"
for METHOD in "${METHODS[@]}"; do
    echo "--------------Running Pretrain Task: $METHOD in Database $DBNAME ------------------"
    python -m ram.pretrain_baseline \
    --tf_cache_dir ./data/rel-avito-tensor-frame \
    --db_name $DBNAME \
    --output_dir ./static\
    --method $METHOD
    echo "-------------Finished Pretrain Task: $METHOD in Database $DBNAME------------------"
    echo
done



DBNAME="stack"
for METHOD in "${METHODS[@]}"; do
    echo "--------------Running Pretrain Task: $METHOD in Database $DBNAME ------------------"
    python -m ram.pretrain_baseline \
    --tf_cache_dir ./data/stack-tensor-frame \
    --db_name $DBNAME \
    --output_dir ./static \
    --method $METHOD
    echo "-------------Finished Pretrain Task: $METHOD in Database $DBNAME------------------"
    echo
done



# RateBeer Database
DBNAME="ratebeer"
for METHOD in "${METHODS[@]}"; do
    echo "--------------Running Pretrain Task: $METHOD in Database $DBNAME ------------------"
    python -m ram.pretrain_baseline \
    --tf_cache_dir ./data/ratebeer-tensor-frame \
    --db_name $DBNAME \
    --output_dir ./static \
    --method $METHOD
    echo "-------------Finished Pretrain Task: $METHOD in Database $DBNAME------------------"
    echo
done

echo -e "\n========================================"
echo "All pretraining baseline experiments completed!"
echo "========================================"
