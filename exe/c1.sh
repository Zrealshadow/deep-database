    python ./cmds/node2vec_baseline.py \
        --tf_cache_dir ./data/f1-tensor-frame \
        --db_name f1 \
        --task_name driver-top3 \
        --channel 128 \
        --batch_size 128 

python ./cmds/generate_table_data.py \
 --dbname f1 \
 --task_name driver-position \
 --sample_size 100000 \
 --table_output_dir ./data/flatten-table


python ./cmds/generate_table_data.py \
 --dbname trial \
 --task_name study-outcome \
 --sample_size 100000 \
 --dfs \
 --selection \
 --table_output_dir ./data/dfs-fs-table


python -m .cmds.ml_baseline \
    --data_dir "./data/dfs-fs-table/hm-user-churn" \
    --method  lgb

python ./cmds/dnn_baseline_table_data.py \
    --data_dir "./data/flatten-table/f1-driver-top3" \
    --model MLP


python ./cmds/dnn_baseline_table_data.py \
    --data_dir "./data/flatten-table/f1-driver-position" \
    --model MLP



python ./cmds/dnn_baseline_table_data.py \
    --data_dir "./data/dfs-fs-table/f1-driver-position" \
    --model MLP



python ./cmds/dnn_baseline_table_data.py \
    --data_dir "./data/dfs-flatten-table/f1-driver-dnf" \
    --model MLP


python ./cmds/generate_table_data.py \
    --dbname f1  \
    --task_name driver-dnf \
    --sample_size 100000  \
    --table_output_dir ./data/dfs-flatten-table \
    --dfs \
    --n_timedelta 1

python ./cmds/generate_table_data.py \
    --dbname f1  \
    --task_name driver-top3 \
    --sample_size 100000  \
    --table_output_dir ./data/dfs-flatten-table \
    --dfs \
    --n_timedelta 1 \
    --n_jobs 4 

python ./cmds/ml_baseline.py \
    --data_dir "./data/dfs-flatten-table/event-user-repeat" \
    --method  lightgbm

python ./cmds/dnn_baseline_table_data.py \
    --data_dir "./data/dfs-flatten-table/f1-driver-dnf" \
    --model MLP



python ./cmds/dnn_baseline_table_data.py \
    --data_dir "./data/dfs-flatten-table/event-user-repeat" \
    --model MLP

python ./cmds/dnn_baseline_table_data.py \
    --data_dir "./data/flatten-table/avito-user-clicks" \
    --model MLP


python ./cmds/dnn_baseline_table_data.py \
    --data_dir "./data/dfs-flatten-table/avito-user-clicks" \
    --model MLP



python ./cmds/dnn_baseline_table_data.py \
    --data_dir "./data/dfs-fs-table/hm-user-churn" \
    --model MLP

python ./cmds/generate_table_data.py \
    --dbname trial  \
    --task_name study-outcome \
    --sample_size 100000  \
    --table_output_dir ./data/dfs-flatten-table \
    --dfs \
    --n_jobs 1


python ./cmds/generate_table_data.py \
    --dbname event  \
    --task_name user-repeat \
    --sample_size 100000  \
    --table_output_dir ./data/dfs-flatten-table \
    --dfs \
    --n_jobs 1\
    --max_features 500 


python ./cmds/generate_table_data.py \
    --dbname event  \
    --task_name user-repeat \
    --sample_size 100000  \
    --table_output_dir ./data/dfs-flatten-table \
    --dfs \
    --n_jobs 1\
    --max_features 500 


python ./cmds/generate_table_data.py \
    --dbname avito  \
    --task_name user-clicks \
    --sample_size 100000  \
    --dfs \
    --table_output_dir ./data/flatten-table 


python ./cmds/generate_table_data.py \
    --dbname ratebeer  \
    --db_cache_dir /home/lingze/.cache/relbench/ratebeer\
    --task_name user-active \
    --sample_size 100000  \
    --dfs \
    --table_output_dir ./data/dfs-flatten-table 


python ./cmds/generate_table_data.py \
    --dbname hm  \
    --task_name user-churn \
    --sample_size 100000  \
    --table_output_dir ./data/flatten-table 

    
# ------------------------ feature selection -----------------
python -m cmds.generate_table_data \
    --dbname trial  \
    --task_name study-outcome \
    --sample_size 100000  \
    --dfs \
    --selection \
    --table_output_dir ./data/dfs-fs-data







export PYTHONPATH=$(pwd)