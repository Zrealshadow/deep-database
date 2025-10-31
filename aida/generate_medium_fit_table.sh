#!/bin/bash
# Generate fit-medium fit table data
export PYTHONPATH=$(pwd)
set -e  # Exit on error

OUTPUT_DIR="./data/fit-medium-table"
SAMPLE_SIZE=100000
MAX_DEPTH=1
N_TIMEDELTA=4

echo "Running Avito tasks (medium-fit)..."
python ./cmds/generate_table_data.py --dbname avito --task_name user-clicks --sample_size $SAMPLE_SIZE --dfs --selection --n_timedelta $N_TIMEDELTA --max_depth $MAX_DEPTH --table_output_dir $OUTPUT_DIR
python ./cmds/generate_table_data.py --dbname avito --task_name ad-ctr --sample_size $SAMPLE_SIZE --dfs --selection --n_timedelta $N_TIMEDELTA --max_depth $MAX_DEPTH --table_output_dir $OUTPUT_DIR

echo "Running RateBeer tasks (medium-fit)..."
python ./cmds/generate_table_data.py --dbname ratebeer --task_name user-active --sample_size $SAMPLE_SIZE --dfs --selection --n_timedelta $N_TIMEDELTA --max_depth $MAX_DEPTH --table_output_dir $OUTPUT_DIR
python ./cmds/generate_table_data.py --dbname ratebeer --task_name place-positive  --sample_size $SAMPLE_SIZE --dfs --selection --n_timedelta $N_TIMEDELTA --max_depth $MAX_DEPTH --table_output_dir $OUTPUT_DIR
python ./cmds/generate_table_data.py --dbname ratebeer --task_name beer-positive --sample_size $SAMPLE_SIZE --dfs --selection --n_timedelta $N_TIMEDELTA --max_depth $MAX_DEPTH --table_output_dir $OUTPUT_DIR

echo "Running Trial tasks (medium-fit)..."
python ./cmds/generate_table_data.py --dbname trial --task_name site-success --sample_size $SAMPLE_SIZE --dfs --selection --n_timedelta $N_TIMEDELTA --max_depth $MAX_DEPTH --table_output_dir $OUTPUT_DIR
python ./cmds/generate_table_data.py --dbname trial --task_name study-adverse --sample_size $SAMPLE_SIZE --dfs --selection --n_timedelta $N_TIMEDELTA --max_depth $MAX_DEPTH --table_output_dir $OUTPUT_DIR
python ./cmds/generate_table_data.py --dbname trial --task_name study-outcome --sample_size $SAMPLE_SIZE --dfs --selection --n_timedelta $N_TIMEDELTA --max_depth $MAX_DEPTH --table_output_dir $OUTPUT_DIR

echo "Running Event tasks (medium-fit)..."
python ./cmds/generate_table_data.py --dbname event --task_name user-attendance --sample_size $SAMPLE_SIZE --dfs --selection --n_timedelta $N_TIMEDELTA --max_depth $MAX_DEPTH --table_output_dir $OUTPUT_DIR
python ./cmds/generate_table_data.py --dbname event --task_name user-repeat --sample_size $SAMPLE_SIZE --dfs --selection --n_timedelta $N_TIMEDELTA --max_depth $MAX_DEPTH --table_output_dir $OUTPUT_DIR

echo "Running HM tasks (medium-fit)..."
python ./cmds/generate_table_data.py --dbname hm --task_name user-churn --sample_size $SAMPLE_SIZE --dfs --selection --n_timedelta $N_TIMEDELTA --max_depth $MAX_DEPTH --table_output_dir $OUTPUT_DIR
python ./cmds/generate_table_data.py --dbname hm --task_name item-sales --sample_size $SAMPLE_SIZE --dfs --selection --n_timedelta $N_TIMEDELTA --max_depth $MAX_DEPTH --table_output_dir $OUTPUT_DIR

echo "All medium-fit tasks completed!"

