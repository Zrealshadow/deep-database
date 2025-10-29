#!/bin/bash
# Hyperopt Baseline: Model Selection + Training Experiments
# Search space: MLP, ResNet, FTTransformer with hyperparameter optimization

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

SCRIPT="./cmd/aida_optuna.py"

# Configuration
N_TRIALS_MLP=100           # MLP has 108 configs
N_TRIALS_RESNET=100        # ResNet has 108 configs
N_TRIALS_FTTRANS=30        # FTTransformer has only 28 configs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Base data directories
BASE_DATA_DIR_FIT_BEST="/home/lingze/embedding_fusion/data/fit-best-table"
BASE_DATA_DIR_FIT_MEDIUM="/home/lingze/embedding_fusion/data/fit-medium-table"
BASE_DATA_DIR_FLATTEN="/home/lingze/embedding_fusion/data/flatten-table"

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
# [1/3] MLP Search Space
#==============================================================================
#
#echo "=========================================="
#echo "[1/3] MLP Search Space"
#echo "=========================================="
#
## Process fit-best-table datasets
#echo "Processing fit-best-table datasets..."
#for DATASET in "${DATASETS_FIT_BEST[@]}"; do
#    echo ""
#    echo "Processing: MLP on ${DATASET} (fit-best-table) (${N_TRIALS_MLP} trials)"
#    python -u ${SCRIPT} \
#        --data_dir "${BASE_DATA_DIR_FIT_BEST}/${DATASET}" \
#        --model MLP \
#        --n_trials ${N_TRIALS_MLP} \
#        --study_name "MLP_${DATASET}_fit-best_${TIMESTAMP}" \
#        --output_csv "${CSV_FILE}"
#
#    if [ $? -eq 0 ]; then
#        echo "✅ MLP on ${DATASET} (fit-best-table) completed"
#    else
#        echo "❌ MLP on ${DATASET} (fit-best-table) failed"
#    fi
#done
#
## Process fit-medium-table datasets
#echo ""
#echo "Processing fit-medium-table datasets..."
#for DATASET in "${DATASETS_FIT_MEDIUM[@]}"; do
#    echo ""
#    echo "Processing: MLP on ${DATASET} (fit-medium-table) (${N_TRIALS_MLP} trials)"
#    python -u ${SCRIPT} \
#        --data_dir "${BASE_DATA_DIR_FIT_MEDIUM}/${DATASET}" \
#        --model MLP \
#        --n_trials ${N_TRIALS_MLP} \
#        --study_name "MLP_${DATASET}_fit-medium_${TIMESTAMP}" \
#        --output_csv "${CSV_FILE}"
#
#    if [ $? -eq 0 ]; then
#        echo "✅ MLP on ${DATASET} (fit-medium-table) completed"
#    else
#        echo "❌ MLP on ${DATASET} (fit-medium-table) failed"
#    fi
#done
#
## Process flatten-table datasets
#echo ""
#echo "Processing flatten-table datasets..."
#for DATASET in "${DATASETS_FLATTEN[@]}"; do
#    echo ""
#    echo "Processing: MLP on ${DATASET} (flatten-table) (${N_TRIALS_MLP} trials)"
#    python -u ${SCRIPT} \
#        --data_dir "${BASE_DATA_DIR_FLATTEN}/${DATASET}" \
#        --model MLP \
#        --n_trials ${N_TRIALS_MLP} \
#        --study_name "MLP_${DATASET}_flatten_${TIMESTAMP}" \
#        --output_csv "${CSV_FILE}"
#
#    if [ $? -eq 0 ]; then
#        echo "✅ MLP on ${DATASET} (flatten-table) completed"
#    else
#        echo "❌ MLP on ${DATASET} (flatten-table) failed"
#    fi
#done
#
#echo ""
#echo "MLP Search Space Completed!"
#echo ""


#==============================================================================
# [2/3] ResNet Search Space
#==============================================================================

echo "=========================================="
echo "[2/3] ResNet Search Space"
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
        echo "Processing: ResNet on ${DATASET} (${DATA_SOURCE}) (${N_TRIALS_RESNET} trials)"
        python -u ${SCRIPT} \
            --data_dir "${DATA_PATH}" \
            --model ResNet \
            --n_trials ${N_TRIALS_RESNET} \
            --study_name "ResNet_${DATASET}_${DATA_SOURCE}_${TIMESTAMP}" \
            --output_csv "${CSV_FILE}"
        
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
# [3/3] FTTransformer Search Space
#==============================================================================

echo "=========================================="
echo "[3/3] FTTransformer Search Space"
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
        echo "Processing: FTTransformer on ${DATASET} (${DATA_SOURCE}) (${N_TRIALS_FTTRANS} trials)"
        python -u ${SCRIPT} \
            --data_dir "${DATA_PATH}" \
            --model FTTransformer \
            --n_trials ${N_TRIALS_FTTRANS} \
            --study_name "FTTransformer_${DATASET}_${DATA_SOURCE}_${TIMESTAMP}" \
            --output_csv "${CSV_FILE}"
        
        if [ $? -eq 0 ]; then
            echo "✅ FTTransformer on ${DATASET} (${DATA_SOURCE}) completed"
        else
            echo "❌ FTTransformer on ${DATASET} (${DATA_SOURCE}) failed"
        fi
    done
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
