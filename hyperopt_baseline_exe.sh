#!/bin/bash
# Hyperopt Baseline: Model Selection + Training Experiments
# Search space: MLP, ResNet, FTTransformer with hyperparameter optimization

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

SCRIPT="./cmd/aida_hyperopt_sh_baseline.py"

# Configuration
N_TRIALS=100
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Base data directory
BASE_DATA_DIR="/home/lingze/embedding_fusion/data/dfs-flatten-table"

# Output directory includes configuration
OUTPUT_DIR="./result_raw_from_server/hyperopt_sh_baseline_n${N_TRIALS}"
LOG_FILE="${OUTPUT_DIR}/log_hyperopt_sh_baseline.txt"
CSV_FILE="${OUTPUT_DIR}/hyperopt_sh_results.csv"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Redirect all output to log file AND console
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=========================================="
echo "Hyperopt Baseline: Model Selection + Training"
echo "=========================================="
echo "Script: ${SCRIPT}"
echo "Trials per experiment: ${N_TRIALS}"
echo "Output: ${OUTPUT_DIR}"
echo "Log: ${LOG_FILE}"
echo "CSV: ${CSV_FILE}"
echo "Started: $(date)"
echo ""

START_TIME=$(date +%s)

# Dataset list (selected from /home/lingze/embedding_fusion/data/dfs-flatten-table)
# Verified to exist on server
DATASETS=(
  event-user-repeat          # event user-repeat ✅
  ratebeer-user-active       # ratebeer user-active ✅
  trial-study-outcome        # trial study-outcome ✅
  avito-user-clicks          # avito user-clicks ✅
  amazon-user-churn          # amazon user-churn (hm user-churn) ✅
  event-user-attendance      # event user-attendance ✅
  ratebeer-beer-positive     # ratebeer beer-positive ✅
  trial-site-success         # trial site-success ✅
  avito-ad-ctr               # avito ad-ctr ✅
  amazon-item-ltv            # amazon item-ltv (hm item-sales) ✅
)


#==============================================================================
# [1/3] MLP Search Space
#==============================================================================

echo "=========================================="
echo "[1/3] MLP Search Space"
echo "=========================================="

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "Processing: MLP on ${DATASET}"
    python -u ${SCRIPT} \
        --data_dir "${BASE_DATA_DIR}/${DATASET}" \
        --model MLP \
        --n_trials ${N_TRIALS} \
        --study_name "MLP_${DATASET}_${TIMESTAMP}" \
        --output_csv "${CSV_FILE}"
    
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
# [2/3] ResNet Search Space
#==============================================================================

echo "=========================================="
echo "[2/3] ResNet Search Space"
echo "=========================================="

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "Processing: ResNet on ${DATASET}"
    python -u ${SCRIPT} \
        --data_dir "${BASE_DATA_DIR}/${DATASET}" \
        --model ResNet \
        --n_trials ${N_TRIALS} \
        --study_name "ResNet_${DATASET}_${TIMESTAMP}" \
        --output_csv "${CSV_FILE}"
    
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
# [3/3] FTTransformer Search Space
#==============================================================================

echo "=========================================="
echo "[3/3] FTTransformer Search Space"
echo "=========================================="

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "Processing: FTTransformer on ${DATASET}"
    python -u ${SCRIPT} \
        --data_dir "${BASE_DATA_DIR}/${DATASET}" \
        --model FTTransformer \
        --n_trials ${N_TRIALS} \
        --study_name "FTTransformer_${DATASET}_${TIMESTAMP}" \
        --output_csv "${CSV_FILE}"
    
    if [ $? -eq 0 ]; then
        echo "✅ FTTransformer on ${DATASET} completed"
    else
        echo "❌ FTTransformer on ${DATASET} failed"
    fi
done

echo ""
echo "FTTransformer Search Space Completed!"
echo ""


#==============================================================================
# Summary
#==============================================================================

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "=========================================="
echo "✅ Hyperopt Baseline Experiments Completed!"
echo "=========================================="
echo "Total time: $((ELAPSED / 3600))h $(((ELAPSED % 3600) / 60))m $((ELAPSED % 60))s"
echo "Results: ${OUTPUT_DIR}"
echo "CSV: ${CSV_FILE}"
echo "Completed: $(date)"
echo "=========================================="
