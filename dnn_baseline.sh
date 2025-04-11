

python ./cmd/dnn_baseline_table_data.py \
    --data_dir ./data/flatten-table/event-user-ignore \
    --model MLP

python ./cmd/dnn_baseline_table_data.py \
    --data_dir ./data/flatten-table/event-user-ignore \
    --model ResNet

python ./cmd/dnn_baseline_table_data.py \
    --data_dir ./data/flatten-table/event-user-ignore \
    --model FTTrans



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