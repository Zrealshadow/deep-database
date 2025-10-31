#!/bin/bash
# Generate fit-low fit table data (flatten baseline without DFS)
export PYTHONPATH=$(pwd)
set -e  # Exit on error

OUTPUT_DIR="./data/fit-low-table"
SAMPLE_SIZE=100000

echo "Running Event tasks (low-fit)..."
python ./cmds/generate_table_data.py --dbname event --task_name user-attendance --sample_size $SAMPLE_SIZE --table_output_dir $OUTPUT_DIR
python ./cmds/generate_table_data.py --dbname event --task_name user-ignore --sample_size $SAMPLE_SIZE --table_output_dir $OUTPUT_DIR
python ./cmds/generate_table_data.py --dbname event --task_name user-repeat --sample_size $SAMPLE_SIZE --table_output_dir $OUTPUT_DIR

echo "Running Stack tasks (low-fit)..."
python ./cmds/generate_table_data.py --dbname stack --task_name user-engagement --sample_size $SAMPLE_SIZE --table_output_dir $OUTPUT_DIR
python ./cmds/generate_table_data.py --dbname stack --task_name user-badge  --sample_size $SAMPLE_SIZE --table_output_dir $OUTPUT_DIR
python ./cmds/generate_table_data.py --dbname stack --task_name post-votes --sample_size $SAMPLE_SIZE --table_output_dir $OUTPUT_DIR

echo "Running Trial tasks (low-fit)..."
python ./cmds/generate_table_data.py --dbname trial --task_name study-outcome --sample_size $SAMPLE_SIZE --table_output_dir $OUTPUT_DIR
python ./cmds/generate_table_data.py --dbname trial --task_name study-adverse --sample_size $SAMPLE_SIZE --table_output_dir $OUTPUT_DIR
python ./cmds/generate_table_data.py --dbname trial --task_name site-success --sample_size $SAMPLE_SIZE --table_output_dir $OUTPUT_DIR

echo "Running Avito tasks (low-fit)..."
python ./cmds/generate_table_data.py --dbname avito --task_name user-visits --sample_size $SAMPLE_SIZE --table_output_dir $OUTPUT_DIR
python ./cmds/generate_table_data.py --dbname avito --task_name user-clicks --sample_size $SAMPLE_SIZE --table_output_dir $OUTPUT_DIR
python ./cmds/generate_table_data.py --dbname avito --task_name ad-ctr --sample_size $SAMPLE_SIZE --table_output_dir $OUTPUT_DIR

echo "Running RateBeer tasks (low-fit)..."
python ./cmds/generate_table_data.py --dbname ratebeer --task_name user-active --sample_size $SAMPLE_SIZE --table_output_dir $OUTPUT_DIR
python ./cmds/generate_table_data.py --dbname ratebeer --task_name place-positive --sample_size $SAMPLE_SIZE --table_output_dir $OUTPUT_DIR
python ./cmds/generate_table_data.py --dbname ratebeer --task_name beer-positive --sample_size $SAMPLE_SIZE --table_output_dir $OUTPUT_DIR

echo "Running F1 tasks (low-fit)..."
python ./cmds/generate_table_data.py --dbname f1 --task_name driver-dnf --sample_size $SAMPLE_SIZE --table_output_dir $OUTPUT_DIR
python ./cmds/generate_table_data.py --dbname f1 --task_name driver-top3 --sample_size $SAMPLE_SIZE --table_output_dir $OUTPUT_DIR

echo "Running HM tasks (low-fit)..."
python ./cmds/generate_table_data.py --dbname hm --task_name user-churn --sample_size $SAMPLE_SIZE --table_output_dir $OUTPUT_DIR
python ./cmds/generate_table_data.py --dbname hm --task_name item-sales --sample_size $SAMPLE_SIZE --table_output_dir $OUTPUT_DIR

echo "All low-fit tasks completed!"
