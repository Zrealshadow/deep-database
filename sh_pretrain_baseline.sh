



#!/bin/bash
export PYTHONPATH=$(pwd)

METHODS=("dgi" "graphcl")

# DBNAME="event"
# for METHOD in "${METHODS[@]}"; do
#     echo "--------------Running Pretrain Task: $METHOD in Database $DBNAME ------------------"
#     python ./cmd/pretrain_baseline.py \
#     --tf_cache_dir ./data/rel-event-tensor-frame \
#     --db_name $DBNAME \
#     --output_dir ./static \
#     --method $METHOD
#     echo "-------------Finished Pretrain Task: $METHOD in Database $DBNAME------------------"
#     echo
# done



DBNAME="trial"
for METHOD in "${METHODS[@]}"; do
    echo "--------------Running Pretrain Task: $METHOD in Database $DBNAME ------------------"
    python ./cmd/pretrain_baseline.py \
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
    python ./cmd/pretrain_baseline.py \
    --tf_cache_dir ./data/rel-avito-tensor-frame \
    --db_name $DBNAME \
    --output_dir ./static\
    --method $METHOD
    echo "-------------Finished Pretrain Task: $METHOD in Database $DBNAME------------------"
    echo
done



# DBNAME="stack"
# for METHOD in "${METHODS[@]}"; do
#     echo "--------------Running Pretrain Task: $METHOD in Database $DBNAME ------------------"
#     python ./cmd/pretrain_baseline.py \
#     --tf_cache_dir ./data/stack-tensor-frame \
#     --data_cache_dir /home/lingze/.cache/relbench/stack \
#     --db_name $DBNAME \
#     --output_dir ./static \
#     --method $METHOD
#     echo "-------------Finished Pretrain Task: $METHOD in Database $DBNAME------------------"
#     echo
# done