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
echo "📊 Experiment Configuration:"
echo "   Datasets: ${#DATASETS[@]}"
echo "   Models: 2 (MLP, ResNet)"
echo "   Total experiments: $((${#DATASETS[@]} * 2))"
echo ""

START_TIME=$(date +%s)

# Dataset list (selected from /home/lingze/embedding_fusion/data/dfs-fs-table, feature-selection)
# Verified to exist on server.
DATASETS=(
  event-user-repeat          # event user-repeat ✅
  ratebeer-user-active       # ratebeer user-active ✅
  trial-study-outcome        # trial study-outcome ✅
  avito-user-clicks          # avito user-clicks ✅
  hm-user-churn              # hm user-churn) ✅
  event-user-attendance      # event user-attendance ✅
  ratebeer-beer-positive     # ratebeer beer-positive ✅
  trial-site-success         # trial site-success ✅
  avito-ad-ctr               # avito ad-ctr ✅
  hm-item-sales              # hm item-sales ✅
)

#==============================================================================
# [1/2] MLP Search Space
#==============================================================================

echo "=========================================="
echo "[1/2] MLP Search Space"
echo "=========================================="

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "Processing: MLP on ${DATASET}"
    python3 -u ${SCRIPT} \
        --data_dir "${BASE_DATA_DIR}/${DATASET}" \
        --space_name "mlp" \
        --output_csv "${CSV_FILE}" \
        --device "cuda:0" \
        --seed 42
    
    if [ $? -eq 0 ]; then
        echo "✅ MLP on ${DATASET} completed"
    else
        echo "❌ MLP on ${DATASET} failed"
    fi
done

echo ""
echo "MLP Search Space Completed!"
echo ""

#==============================================================================
# [2/2] ResNet Search Space
#==============================================================================

echo "=========================================="
echo "[2/2] ResNet Search Space"
echo "=========================================="

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "Processing: ResNet on ${DATASET}"
    python3 -u ${SCRIPT} \
        --data_dir "${BASE_DATA_DIR}/${DATASET}" \
        --space_name "resnet" \
        --output_csv "${CSV_FILE}" \
        --device "cuda:0" \
        --seed 42
    
    if [ $? -eq 0 ]; then
        echo "✅ ResNet on ${DATASET} completed"
    else
        echo "❌ ResNet on ${DATASET} failed"
    fi
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
