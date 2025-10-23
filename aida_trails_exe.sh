#!/bin/bash
# AIDA Trails: Evolutionary Algorithm for Model Selection Experiments
# Search space: MLP, ResNet with evolutionary algorithm

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

SCRIPT="./cmd/aida_trails.py"

# Configuration
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Base data directory
BASE_DATA_DIR="/home/lingze/embedding_fusion/data/dfs-fs-table"

# Output directory
OUTPUT_DIR="./result_raw_from_server/aida_trails"
LOG_FILE="${OUTPUT_DIR}/log_aida_trails.txt"
CSV_FILE="${OUTPUT_DIR}/aida_trails_results.csv"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Redirect all output to log file AND console
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=========================================="
echo "AIDA Trails: Evolutionary Algorithm for Model Selection"
echo "=========================================="
echo "Script: ${SCRIPT}"
echo "Output: ${OUTPUT_DIR}"
echo "Log: ${LOG_FILE}"
echo "CSV: ${CSV_FILE}"
echo "Started: $(date)"
echo ""

START_TIME=$(date +%s)

# Dataset list (using sample data for testing)
DATASETS=(
    avito-user-clicks          # avito user-clicks ✅
)

# Model types
MODELS=(
    mlp                        # MLP with evolutionary algorithm
    resnet                     # ResNet with evolutionary algorithm
)

#==============================================================================
# Run AIDA Trails for each dataset and model combination
#==============================================================================

for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        echo ""
        echo "=========================================="
        echo "Processing: ${MODEL} on ${DATASET}"
        echo "=========================================="
        
        python3 -u ${SCRIPT} \
            --data_dir "${BASE_DATA_DIR}/${DATASET}" \
            --space_name "${MODEL}" \
            --output_csv "${CSV_FILE}" \
            --device "cuda:0" \
            --seed 42
        
        if [ $? -eq 0 ]; then
            echo "✅ ${MODEL} on ${DATASET} completed"
        else
            echo "❌ ${MODEL} on ${DATASET} failed"
        fi
    done
done

#==============================================================================
# Summary
#==============================================================================

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "✅ AIDA Trails Experiments Completed!"
echo "=========================================="
echo "Total time: $((ELAPSED / 3600))h $(((ELAPSED % 3600) / 60))m $((ELAPSED % 60))s"
echo "Results: ${OUTPUT_DIR}"
echo "CSV: ${CSV_FILE}"
echo "Completed: $(date)"
echo "=========================================="
