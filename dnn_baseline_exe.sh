

python ./cmd/dnn_baseline_table_data.py \
    --data_dir ./data/flatten-table/event-user-ignore \
    --model MLP


python ./cmd/dnn_baseline_table_data.py \
    --data_dir ./data/flatten-table/event-user-repeat\
    --model MLP

python ./cmd/dnn_baseline_table_data.py \
    --data_dir ./data/flatten-table/event-user-repeat\
    --model ResNet

python ./cmd/dnn_baseline_table_data.py \
    --data_dir ./data/flatten-table/event-user-repeat\
    --model FTTrans


python ./cmd/dnn_baseline_table_data.py \
    --data_dir ./data/flatten-table/stack-user-engagement\
    --model MLP


python ./cmd/dnn_baseline_table_data.py \
    --data_dir ./data/flatten-table/stack-user-badge\
    --model MLP


python ./cmd/dnn_baseline_table_data.py \
    --data_dir ./data/flatten-table/trial-study-outcome\
    --model MLP

python ./cmd/dnn_baseline_table_data.py \
    --data_dir ./data/flatten-table/avito-user-visits\
    --model MLP


python ./cmd/dnn_baseline_table_data.py \
    --data_dir ./data/flatten-table/avito-user-clicks\
    --model MLP


# --------------------- Regression

python ./cmd/dnn_baseline_table_data.py \
    --data_dir ./data/flatten-table/avito-ad-ctr\
    --model MLP 


python ./cmd/dnn_baseline_table_data.py \
    --data_dir ./data/flatten-table/event-user-attendance\
    --model MLP


python ./cmd/dnn_baseline_table_data.py \
    --data_dir ./data/flatten-table/trial-study-adverse\
    --model MLP

python ./cmd/dnn_baseline_table_data.py \
    --data_dir ./data/flatten-table/trial-site-success\
    --model MLP

python ./cmd/dnn_baseline_table_data.py \
    --data_dir ./data/flatten-table/stack-post-votes\
    --model MLP