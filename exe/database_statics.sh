#!/bin/bash

export PYTHONPATH=$(pwd)

db_list=(
    "event"
    "stack"
    "trial"
    "avito"
    "ratebeer"
    "f1"
    "hm"
    "amazon"
)

echo "Running database statics ..."

for db_name in "${db_list[@]}"; do
    echo "Processing database: $db_name"
    python ./cmds/database_statics.py --db_name "$db_name"
done

echo "All database statics completed!"
