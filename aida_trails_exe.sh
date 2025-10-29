#!/bin/bash
# AIDA Trails: Evolutionary Algorithm for Model Selection Experiments
# Search space: MLP, ResNet with evolutionary algorithm

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

SCRIPT="${SCRIPT_DIR}/cmds/aida_trails.py"

# Configuration
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Base data directories
BASE_DATA_DIR_FIT_BEST="/home/lingze/embedding_fusion/data/fit-best-table"
BASE_DATA_DIR_FIT_MEDIUM="/home/lingze/embedding_fusion/data/fit-medium-table"
BASE_DATA_DIR_FLATTEN="/home/lingze/embedding_fusion/data/flatten-table"

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
echo "📊 Experiment Configuration:"
echo "   Datasets: ${#DATASETS[@]}"
echo "   Data sources: ${#DATA_DIRS[@]} (fit-best-table, fit-medium-table, flatten-table)"
echo "   Models: 1 (ResNet only - MLP commented out)"
TOTAL_EXPERIMENTS=$((${#DATASETS[@]} * ${#DATA_DIRS[@]} * 1))
echo "   Total experiments: ${TOTAL_EXPERIMENTS}"
echo ""

START_TIME=$(date +%s)

# Selected datasets - Only the 10 core datasets
DATASETS=(
  avito-ad-ctr
  avito-user-clicks
  event-user-attendance
  event-user-repeat
  hm-item-sales
  hm-user-churn
  ratebeer-beer-positive
  ratebeer-user-active
  trial-site-success
  trial-study-outcome
)

# Data source directories
DATA_DIRS=(
  "fit-best-table"
  "fit-medium-table" 
  "flatten-table"
)

#==============================================================================
# [1/2] MLP Search Space - COMMENTED OUT
#==============================================================================
#
#echo "=========================================="
#echo "[1/2] MLP Search Space"
#echo "=========================================="
#
## Process all datasets from all data sources
#for DATA_SOURCE in "${DATA_DIRS[@]}"; do
#    echo ""
#    echo "Processing ${DATA_SOURCE} datasets..."
#    
#    for DATASET in "${DATASETS[@]}"; do
#        # Check if dataset exists in this data source
#        DATA_PATH="/home/lingze/embedding_fusion/data/${DATA_SOURCE}/${DATASET}"
#        if [ ! -d "${DATA_PATH}" ]; then
#            echo "⚠️  Skipping ${DATASET} (not found in ${DATA_SOURCE})"
#            continue
#        fi
#        
#        echo ""
#        echo "Processing: MLP on ${DATASET} (${DATA_SOURCE})"
#        python3 -u ${SCRIPT} \
#            --data_dir "${DATA_PATH}" \
#            --space_name "mlp" \
#            --output_csv "${CSV_FILE}" \
#            --device "cuda:0" \
#            --seed 42
#        
#        if [ $? -eq 0 ]; then
#            echo "✅ MLP on ${DATASET} (${DATA_SOURCE}) completed"
#        else
#            echo "❌ MLP on ${DATASET} (${DATA_SOURCE}) failed"
#        fi
#    done
#done
#
#echo ""
#echo "MLP Search Space Completed!"
#echo ""

#==============================================================================
# [2/2] ResNet Search Space
#==============================================================================

echo "=========================================="
echo "[2/2] ResNet Search Space"
echo "=========================================="

# Process all datasets from all data sources
for DATA_SOURCE in "${DATA_DIRS[@]}"; do
    echo ""
    echo "Processing ${DATA_SOURCE} datasets..."
    
    for DATASET in "${DATASETS[@]}"; do
        # Check if dataset exists in this data source
        DATA_PATH="/home/lingze/embedding_fusion/data/${DATA_SOURCE}/${DATASET}"
        if [ ! -d "${DATA_PATH}" ]; then
            echo "⚠️  Skipping ${DATASET} (not found in ${DATA_SOURCE})"
            continue
        fi
        
        echo ""
        echo "Processing: ResNet on ${DATASET} (${DATA_SOURCE})"
        python3 -u ${SCRIPT} \
            --data_dir "${DATA_PATH}" \
            --space_name "resnet" \
            --output_csv "${CSV_FILE}" \
            --device "cuda:0" \
            --seed 42
        
        if [ $? -eq 0 ]; then
            echo "✅ ResNet on ${DATASET} (${DATA_SOURCE}) completed"
        else
            echo "❌ ResNet on ${DATASET} (${DATA_SOURCE}) failed"
        fi
    done
done

echo ""
echo "ResNet Search Space Completed!"
echo ""

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
