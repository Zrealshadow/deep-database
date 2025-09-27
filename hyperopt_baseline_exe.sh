#!/bin/bash

# Set environment variables to avoid OpenMP conflicts
export PYTHONPATH=$(pwd)

# Stop the script if any command fails
set -e

# Create results directory if it doesn't exist
mkdir -p results/hyperopt

# Create a single comprehensive results file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="results/hyperopt/hyperopt_results_${TIMESTAMP}.txt"

echo "Hyperopt Baseline Results - $(date)" > "$RESULTS_FILE"
echo "===============================================" >> "$RESULTS_FILE"
echo "Models: ${MODELS[*]}" >> "$RESULTS_FILE"
echo "Data directories: ${#DATA_DIR_LIST[@]}" >> "$RESULTS_FILE"
echo "Trials per experiment: $N_TRIALS" >> "$RESULTS_FILE"
echo "Total experiments: $((${#MODELS[@]} * ${#DATA_DIR_LIST[@]}))" >> "$RESULTS_FILE"
echo "===============================================" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# Define models to test
MODELS=("ResNet" "FTTransformer")

# Define data directories
DATA_DIR_LIST=(
    "avito-ad-ctr"
    "event-user-ignore"
    "ratebeer-beer-positive"
    "ratebeer-user-active"
    "trial-site-success"
    "trial-study-outcome"
    "avito-user-clicks"
    "event-user-attendance"
    "event-user-repeat"
    "avito-user-visits"
    "ratebeer-place-positive"
    "trial-study-adverse"
)

# Number of trials for hyperparameter optimization
N_TRIALS=100

echo "Starting Hyperopt Baseline Experiments..."
echo "========================================"
echo "Models: ${MODELS[*]}"
echo "Data directories: ${#DATA_DIR_LIST[@]} directories"
echo "Trials per experiment: $N_TRIALS"
echo "Results will be saved to: $RESULTS_FILE"
echo "========================================"
echo

# Run hyperparameter optimization for each model and data directory
for MODEL in "${MODELS[@]}"; do
    echo "=============== Running Hyperopt for Model: $MODEL ==============="
    echo "=============== Running Hyperopt for Model: $MODEL ===============" >> "$RESULTS_FILE"
    
    for DATA_DIR in "${DATA_DIR_LIST[@]}"; do
        echo "Processing: $DATA_DIR with $MODEL"
        echo "Processing: $DATA_DIR with $MODEL" >> "$RESULTS_FILE"
        
        # Create study name
        STUDY_NAME="${MODEL}_${DATA_DIR}_${TIMESTAMP}"
        
        echo "Study name: $STUDY_NAME"
        echo "Data directory: /home/lingze/embedding_fusion/data/dfs-flatten-table/$DATA_DIR"
        echo "----------------------------------------"
        
        # Run the hyperparameter optimization and capture output
        echo "----------------------------------------" >> "$RESULTS_FILE"
        echo "EXPERIMENT: $MODEL on $DATA_DIR" >> "$RESULTS_FILE"
        echo "Study: $STUDY_NAME" >> "$RESULTS_FILE"
        echo "Data: /home/lingze/embedding_fusion/data/dfs-flatten-table/$DATA_DIR" >> "$RESULTS_FILE"
        echo "----------------------------------------" >> "$RESULTS_FILE"

        python ./cmd/hyperopt_baseline.py \
            --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/$DATA_DIR" \
            --model "$MODEL" \
            --n_trials "$N_TRIALS" \
            --study_name "$STUDY_NAME" \
            >> "$RESULTS_FILE" 2>&1
        
        # Check if the command was successful
        if [ $? -eq 0 ]; then
            echo "✅ Successfully completed: $MODEL on $DATA_DIR"
            echo "✅ SUCCESS: $MODEL on $DATA_DIR" >> "$RESULTS_FILE"
        else
            echo "❌ Failed: $MODEL on $DATA_DIR"
            echo "❌ FAILED: $MODEL on $DATA_DIR" >> "$RESULTS_FILE"
        fi
        
        # Clean up the database file to save space
        if [ -f "studies/${STUDY_NAME}.db" ]; then
            rm "studies/${STUDY_NAME}.db"
        fi
        
        echo "----------------------------------------"
        echo "----------------------------------------" >> "$RESULTS_FILE"
        echo "" >> "$RESULTS_FILE"
    done
    
    echo "=============== Finished Model: $MODEL ==============="
    echo "=============== Finished Model: $MODEL ===============" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"
done

echo "========================================"
echo "All hyperparameter optimization experiments completed!"
echo "All results saved to: $RESULTS_FILE"
echo "Note: Database files were cleaned up to save space"
echo "========================================"

echo "========================================" >> "$RESULTS_FILE"
echo "EXPERIMENT SUMMARY" >> "$RESULTS_FILE"
echo "========================================" >> "$RESULTS_FILE"
echo "Completed at: $(date)" >> "$RESULTS_FILE"
echo "Models tested: ${MODELS[*]}" >> "$RESULTS_FILE"
echo "Data directories: ${#DATA_DIR_LIST[@]}" >> "$RESULTS_FILE"
echo "Trials per experiment: $N_TRIALS" >> "$RESULTS_FILE"
echo "Total experiments: $((${#MODELS[@]} * ${#DATA_DIR_LIST[@]}))" >> "$RESULTS_FILE"
echo "========================================" >> "$RESULTS_FILE"

echo "All results saved to: $RESULTS_FILE"
