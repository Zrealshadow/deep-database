#!/bin/bash
export PYTHONPATH=$(pwd)

data_dir_root="./data"

type_dir_list=(
    "flatten-table"
    "fit-medium-table"
    "fit-best-table"
)

data_list=(
    "avito-user-clicks"
    # "avito-ad-ctr"
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



echo "Running tabular data statics ..."

for type_dir in "${type_dir_list[@]}"; do
    for data_name in "${data_list[@]}"; do
        data_dir="${data_dir_root}/${type_dir}/${data_name}"
        python ./cmds/tabular_statics.py --data_dir "${data_dir}"
    done
done

echo "All tabular data statics completed!"