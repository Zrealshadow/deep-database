#!/bin/bash
# Leva table generation script

# Configuration
output_dir="./data/leva-table"
verbose_flag="--verbose"

# Dataset mappings: dataset_name -> db_name task_name
declare -A dataset_map=(
    ["avito-user-clicks"]="avito user-clicks"
    ["avito-ad-ctr"]="avito ad-ctr"
    ["event-user-repeat"]="event user-repeat"
    ["event-user-attendance"]="event user-attendance"
    ["ratebeer-beer-positive"]="ratebeer beer-positive"
    ["ratebeer-place-positive"]="ratebeer place-positive"
    ["ratebeer-user-active"]="ratebeer user-active"
    ["trial-site-success"]="trial site-success"
    ["trial-study-outcome"]="trial study-outcome"
    ["hm-item-sales"]="hm item-sales"
    ["hm-user-churn"]="hm user-churn"
)

# List of datasets to process
data_list=(
    # "avito-user-clicks"
    # "avito-ad-ctr"
    # "event-user-repeat"
    # "event-user-attendance"
    # "ratebeer-beer-positive"
    # "ratebeer-place-positive"
    # "ratebeer-user-active"
    "trial-site-success"
    "trial-study-outcome"
    "hm-item-sales"
    "hm-user-churn"
)

# Function to generate leva table
generate_leva_table() {
    local dataset=$1
    local mapping="${dataset_map[$dataset]}"

    if [ -z "$mapping" ]; then
        echo "Error: No mapping found for dataset: $dataset"
        return 1
    fi

    # Split mapping into db_name and task_name
    read -r db_name task_name <<< "$mapping"

    echo "Processing: $dataset (db=$db_name, task=$task_name)"

    python -m leva.generate_leva_table \
        --db_name "$db_name" \
        --task_name "$task_name" \
        --table_output_dir "$output_dir" \
        $verbose_flag
}

# Loop through all datasets
for dataset in "${data_list[@]}"; do
    generate_leva_table "$dataset"
done

echo "All Leva table generation completed!"