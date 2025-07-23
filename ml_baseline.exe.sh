#!/bin/bash
export PYTHONPATH=$(pwd)

METHODS=("lgb" "catboost")
DATA_DIR_LIST=(
    "avito-ad-ctr"
    "event-user-ignore"
    "ratebeer-beer-positive"
    "ratebeer-user-active"
    "stack-user-badge"
    "trial-site-success"
    "trial-study-outcome"
    "avito-user-clicks"
    "event-user-attendance"
    "event-user-repeat"
    "avito-user-visits"
    "ratebeer-place-positive"
    "stack-post-votes"
    "stack-user-engagement"
    "trial-study-adverse"
)


for METHOD in "${METHODS[@]}"; do
    echo "--------------Running ML Baseline Task: $METHOD ------------------"
    for DATA_DIR in "${DATA_DIR_LIST[@]}"; do
        echo "Using data directory: $DATA_DIR"
        python ./cmd/ml_baseline.py \
        --data_dir "./data/flatten-table/$DATA_DIR" \
        --method $METHOD 
    done
    echo "-------------Finished ML Baseline Task: $METHOD------------------"
    echo
done
