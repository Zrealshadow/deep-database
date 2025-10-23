#!/bin/bash
# Hyperopt Baseline: Model Selection + Training Experiments
# Search space: MLP, ResNet, FTTransformer with hyperparameter optimization

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

SCRIPT="./cmd/aida_fit_best_baseline.py"

# Configuration
N_TRIALS_MLP=100           # MLP has 108 configs
N_TRIALS_RESNET=100        # ResNet has 108 configs
N_TRIALS_FTTRANS=30        # FTTransformer has only 28 configs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Base data directory
BASE_DATA_DIR="/home/lingze/embedding_fusion/data/dfs-fs-table"

# Output directory
OUTPUT_DIR="./result_raw_from_server/hyperopt_sh_baseline"
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
echo "Trials: MLP=${N_TRIALS_MLP}, ResNet=${N_TRIALS_RESNET}, FTTransformer=${N_TRIALS_FTTRANS}"
echo "Output: ${OUTPUT_DIR}"
echo "Log: ${LOG_FILE}"
echo "CSV: ${CSV_FILE}"
echo "Started: $(date)"
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
# [1/3] MLP Search Space
#==============================================================================
#
#echo "=========================================="
#echo "[1/3] MLP Search Space"
#echo "=========================================="
#
#for DATASET in "${DATASETS[@]}"; do
#    echo ""
#    echo "Processing: MLP on ${DATASET} (${N_TRIALS_MLP} trials)"
#    python -u ${SCRIPT} \
#        --data_dir "${BASE_DATA_DIR}/${DATASET}" \
#        --model MLP \
#        --n_trials ${N_TRIALS_MLP} \
#        --study_name "MLP_${DATASET}_${TIMESTAMP}" \
#        --output_csv "${CSV_FILE}"
#
#    if [ $? -eq 0 ]; then
#        echo "✅ MLP on ${DATASET} completed"
#    else
#        echo "❌ MLP on ${DATASET} failed"
#    fi
#done
#
#echo ""
#echo "MLP Search Space Completed!"
#echo ""


#==============================================================================
# [2/3] ResNet Search Space
#==============================================================================
#
#echo "=========================================="
#echo "[2/3] ResNet Search Space"
#echo "=========================================="
#
#for DATASET in "${DATASETS[@]}"; do
#    echo ""
#    echo "Processing: ResNet on ${DATASET} (${N_TRIALS_RESNET} trials)"
#    python -u ${SCRIPT} \
#        --data_dir "${BASE_DATA_DIR}/${DATASET}" \
#        --model ResNet \
#        --n_trials ${N_TRIALS_RESNET} \
#        --study_name "ResNet_${DATASET}_${TIMESTAMP}" \
#        --output_csv "${CSV_FILE}"
#
#    if [ $? -eq 0 ]; then
#        echo "✅ ResNet on ${DATASET} completed"
#    else
#        echo "❌ ResNet on ${DATASET} failed"
#    fi
#done
#
#echo ""
#echo "ResNet Search Space Completed!"
#echo ""


#==============================================================================
# [3/3] FTTransformer Search Space
#==============================================================================

echo "=========================================="
echo "[3/3] FTTransformer Search Space"
echo "=========================================="

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "Processing: FTTransformer on ${DATASET} (${N_TRIALS_FTTRANS} trials)"
    python -u ${SCRIPT} \
        --data_dir "${BASE_DATA_DIR}/${DATASET}" \
        --model FTTransformer \
        --n_trials ${N_TRIALS_FTTRANS} \
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
