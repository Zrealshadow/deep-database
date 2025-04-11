#!/bin/bash

set -e  # Exit on error

OUTPUT_DIR="./data/flatten-table"
CACHE_DIR="/home/lingze/.cache/relbench/stack"
SAMPLE_SIZE=100000

echo "Running Event tasks..."
python ./cmd/generate_table_data.py --dbname event --task_name user-attendance --sample_size $SAMPLE_SIZE --table_output_dir $OUTPUT_DIR
python ./cmd/generate_table_data.py --dbname event --task_name user-ignore --sample_size $SAMPLE_SIZE --table_output_dir $OUTPUT_DIR
python ./cmd/generate_table_data.py --dbname event --task_name user-repeat --sample_size $SAMPLE_SIZE --table_output_dir $OUTPUT_DIR

echo "Running Stack tasks..."
python ./cmd/generate_table_data.py --dbname stack --task_name user-engagement --db_cache_dir $CACHE_DIR --sample_size $SAMPLE_SIZE --table_output_dir $OUTPUT_DIR
python ./cmd/generate_table_data.py --dbname stack --task_name user-badge --db_cache_dir $CACHE_DIR --sample_size $SAMPLE_SIZE --table_output_dir $OUTPUT_DIR
python ./cmd/generate_table_data.py --dbname stack --task_name post-votes --db_cache_dir $CACHE_DIR --sample_size $SAMPLE_SIZE --table_output_dir $OUTPUT_DIR

echo "Running Trial tasks..."
python ./cmd/generate_table_data.py --dbname trial --task_name study-outcome --sample_size $SAMPLE_SIZE --table_output_dir $OUTPUT_DIR
python ./cmd/generate_table_data.py --dbname trial --task_name study-adverse --sample_size $SAMPLE_SIZE --table_output_dir $OUTPUT_DIR
python ./cmd/generate_table_data.py --dbname trial --task_name site-success --sample_size $SAMPLE_SIZE --table_output_dir $OUTPUT_DIR

echo "Running Avito tasks..."
python ./cmd/generate_table_data.py --dbname avito --task_name user-visits --sample_size $SAMPLE_SIZE --table_output_dir $OUTPUT_DIR
python ./cmd/generate_table_data.py --dbname avito --task_name user-clicks --sample_size $SAMPLE_SIZE --table_output_dir $OUTPUT_DIR
python ./cmd/generate_table_data.py --dbname avito --task_name ad-ctr --sample_size $SAMPLE_SIZE --table_output_dir $OUTPUT_DIR

echo "✅ All tasks completed!"
