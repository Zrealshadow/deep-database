#!/bin/bash
export PYTHONPATH=$(pwd)
set -e

mkdir -p results/hyperopt
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="results/hyperopt/hyperopt_results_${TIMESTAMP}.txt"

echo "Hyperopt Baseline Results - $(date)" >"$RESULTS_FILE"
echo "===============================================" >>"$RESULTS_FILE"
echo "Trials per experiment: 100" >>"$RESULTS_FILE"
echo "===============================================" >>"$RESULTS_FILE"
echo "" >>"$RESULTS_FILE"

python ./cmd/hyperopt_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/avito-ad-ctr" --model ResNet --n_trials 100 --study_name "ResNet_avito-ad-ctr_${TIMESTAMP}" >>"$RESULTS_FILE" 2>&1
python ./cmd/hyperopt_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/event-user-ignore" --model ResNet --n_trials 100 --study_name "ResNet_event-user-ignore_${TIMESTAMP}" >>"$RESULTS_FILE" 2>&1
python ./cmd/hyperopt_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/ratebeer-beer-positive" --model ResNet --n_trials 100 --study_name "ResNet_ratebeer-beer-positive_${TIMESTAMP}" >>"$RESULTS_FILE" 2>&1
python ./cmd/hyperopt_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/ratebeer-user-active" --model ResNet --n_trials 100 --study_name "ResNet_ratebeer-user-active_${TIMESTAMP}" >>"$RESULTS_FILE" 2>&1
python ./cmd/hyperopt_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/trial-site-success" --model ResNet --n_trials 100 --study_name "ResNet_trial-site-success_${TIMESTAMP}" >>"$RESULTS_FILE" 2>&1
python ./cmd/hyperopt_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/trial-study-outcome" --model ResNet --n_trials 100 --study_name "ResNet_trial-study-outcome_${TIMESTAMP}" >>"$RESULTS_FILE" 2>&1
python ./cmd/hyperopt_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/avito-user-clicks" --model ResNet --n_trials 100 --study_name "ResNet_avito-user-clicks_${TIMESTAMP}" >>"$RESULTS_FILE" 2>&1
python ./cmd/hyperopt_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/event-user-attendance" --model ResNet --n_trials 100 --study_name "ResNet_event-user-attendance_${TIMESTAMP}" >>"$RESULTS_FILE" 2>&1
python ./cmd/hyperopt_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/event-user-repeat" --model ResNet --n_trials 100 --study_name "ResNet_event-user-repeat_${TIMESTAMP}" >>"$RESULTS_FILE" 2>&1
python ./cmd/hyperopt_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/avito-user-visits" --model ResNet --n_trials 100 --study_name "ResNet_avito-user-visits_${TIMESTAMP}" >>"$RESULTS_FILE" 2>&1
python ./cmd/hyperopt_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/ratebeer-place-positive" --model ResNet --n_trials 100 --study_name "ResNet_ratebeer-place-positive_${TIMESTAMP}" >>"$RESULTS_FILE" 2>&1
python ./cmd/hyperopt_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/trial-study-adverse" --model ResNet --n_trials 100 --study_name "ResNet_trial-study-adverse_${TIMESTAMP}" >>"$RESULTS_FILE" 2>&1

python ./cmd/hyperopt_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/avito-ad-ctr" --model FTTransformer --n_trials 100 --study_name "FTTransformer_avito-ad-ctr_${TIMESTAMP}" >>"$RESULTS_FILE" 2>&1
python ./cmd/hyperopt_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/event-user-ignore" --model FTTransformer --n_trials 100 --study_name "FTTransformer_event-user-ignore_${TIMESTAMP}" >>"$RESULTS_FILE" 2>&1
python ./cmd/hyperopt_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/ratebeer-beer-positive" --model FTTransformer --n_trials 100 --study_name "FTTransformer_ratebeer-beer-positive_${TIMESTAMP}" >>"$RESULTS_FILE" 2>&1
python ./cmd/hyperopt_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/ratebeer-user-active" --model FTTransformer --n_trials 100 --study_name "FTTransformer_ratebeer-user-active_${TIMESTAMP}" >>"$RESULTS_FILE" 2>&1
python ./cmd/hyperopt_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/trial-site-success" --model FTTransformer --n_trials 100 --study_name "FTTransformer_trial-site-success_${TIMESTAMP}" >>"$RESULTS_FILE" 2>&1
python ./cmd/hyperopt_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/trial-study-outcome" --model FTTransformer --n_trials 100 --study_name "FTTransformer_trial-study-outcome_${TIMESTAMP}" >>"$RESULTS_FILE" 2>&1
python ./cmd/hyperopt_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/avito-user-clicks" --model FTTransformer --n_trials 100 --study_name "FTTransformer_avito-user-clicks_${TIMESTAMP}" >>"$RESULTS_FILE" 2>&1
python ./cmd/hyperopt_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/event-user-attendance" --model FTTransformer --n_trials 100 --study_name "FTTransformer_event-user-attendance_${TIMESTAMP}" >>"$RESULTS_FILE" 2>&1
python ./cmd/hyperopt_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/event-user-repeat" --model FTTransformer --n_trials 100 --study_name "FTTransformer_event-user-repeat_${TIMESTAMP}" >>"$RESULTS_FILE" 2>&1
python ./cmd/hyperopt_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/avito-user-visits" --model FTTransformer --n_trials 100 --study_name "FTTransformer_avito-user-visits_${TIMESTAMP}" >>"$RESULTS_FILE" 2>&1
python ./cmd/hyperopt_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/ratebeer-place-positive" --model FTTransformer --n_trials 100 --study_name "FTTransformer_ratebeer-place-positive_${TIMESTAMP}" >>"$RESULTS_FILE" 2>&1
python ./cmd/hyperopt_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/trial-study-adverse" --model FTTransformer --n_trials 100 --study_name "FTTransformer_trial-study-adverse_${TIMESTAMP}" >>"$RESULTS_FILE" 2>&1
