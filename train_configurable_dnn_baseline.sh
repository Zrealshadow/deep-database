#!/bin/bash

mkdir -p logs

# ResNet
python ./cmd/dnn_configurable_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/avito-ad-ctr" --model ResNet --channels 512 --num_layers 2 --dropout_prob 0.5 --normalization batch_norm > logs/ResNet_avito-ad-ctr.log 2>&1 &

python ./cmd/dnn_configurable_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/avito-ad-ctr-dfs-depth3-feat1000-tw-1" --model ResNet --channels 64 --num_layers 5 --dropout_prob 0.4 --normalization batch_norm > logs/ResNet_avito-ad-ctr-dfs.log 2>&1 &

python ./cmd/dnn_configurable_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/avito-user-clicks" --model ResNet --channels 64 --num_layers 4 --dropout_prob 0.4 --normalization layer_norm > logs/ResNet_avito-user-clicks.log 2>&1 &

python ./cmd/dnn_configurable_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/event-user-attendance" --model ResNet --channels 64 --num_layers 5 --dropout_prob 0.5 --normalization none > logs/ResNet_event-attendance.log 2>&1 &

python ./cmd/dnn_configurable_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/event-user-ignore" --model ResNet --channels 128 --num_layers 3 --dropout_prob 0.2 --normalization batch_norm > logs/ResNet_event-ignore.log 2>&1 &

python ./cmd/dnn_configurable_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/event-user-repeat" --model ResNet --channels 256 --num_layers 3 --dropout_prob 0.3 --normalization batch_norm > logs/ResNet_event-repeat.log 2>&1 &

python ./cmd/dnn_configurable_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/f1-driver-dnf" --model ResNet --channels 128 --num_layers 6 --dropout_prob 0.3 --normalization layer_norm > logs/ResNet_f1-dnf.log 2>&1 &

python ./cmd/dnn_configurable_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/f1-driver-top3" --model ResNet --channels 256 --num_layers 5 --dropout_prob 0.4 --normalization none > logs/ResNet_f1-top3.log 2>&1 &

python ./cmd/dnn_configurable_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/ratebeer-beer-positive" --model ResNet --channels 512 --num_layers 3 --dropout_prob 0.4 --normalization batch_norm > logs/ResNet_ratebeer-beer.log 2>&1 &

python ./cmd/dnn_configurable_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/ratebeer-place-positive" --model ResNet --channels 256 --num_layers 5 --dropout_prob 0.5 --normalization batch_norm > logs/ResNet_ratebeer-place.log 2>&1 &

python ./cmd/dnn_configurable_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/ratebeer-place-positive-dfs-depth3-feat1000-tw-1" --model ResNet --channels 64 --num_layers 6 --dropout_prob 0.1 --normalization layer_norm > logs/ResNet_ratebeer-place-dfs.log 2>&1 &

python ./cmd/dnn_configurable_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/ratebeer-user-active" --model ResNet --channels 256 --num_layers 6 --dropout_prob 0.2 --normalization batch_norm > logs/ResNet_ratebeer-user.log 2>&1 &

python ./cmd/dnn_configurable_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/trial-site-success" --model ResNet --channels 64 --num_layers 3 --dropout_prob 0.1 --normalization batch_norm > logs/ResNet_trial-site.log 2>&1 &

python ./cmd/dnn_configurable_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/trial-study-adverse" --model ResNet --channels 512 --num_layers 4 --dropout_prob 0.4 --normalization layer_norm > logs/ResNet_trial-adverse.log 2>&1 &

python ./cmd/dnn_configurable_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/trial-study-outcome" --model ResNet --channels 64 --num_layers 3 --dropout_prob 0.1 --normalization batch_norm > logs/ResNet_trial-outcome.log 2>&1 &


# FTTransformer
python ./cmd/dnn_configurable_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/avito-ad-ctr" --model FTTransformer --channels 64 --num_layers 6 > logs/FTTransformer_avito-ad-ctr.log 2>&1 &

python ./cmd/dnn_configurable_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/avito-ad-ctr-dfs-depth3-feat1000-tw-1" --model FTTransformer --channels 64 --num_layers 5 > logs/FTTransformer_avito-ad-ctr-dfs.log 2>&1 &

python ./cmd/dnn_configurable_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/avito-user-clicks" --model FTTransformer --channels 256 --num_layers 2 > logs/FTTransformer_avito-user-clicks.log 2>&1 &

python ./cmd/dnn_configurable_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/event-user-attendance" --model FTTransformer --channels 128 --num_layers 5 > logs/FTTransformer_event-attendance.log 2>&1 &

python ./cmd/dnn_configurable_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/event-user-ignore" --model FTTransformer --channels 512 --num_layers 2 > logs/FTTransformer_event-ignore.log 2>&1 &

python ./cmd/dnn_configurable_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/event-user-repeat" --model FTTransformer --channels 128 --num_layers 2 > logs/FTTransformer_event-repeat.log 2>&1 &

python ./cmd/dnn_configurable_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/f1-driver-dnf" --model FTTransformer --channels 256 --num_layers 2 > logs/FTTransformer_f1-dnf.log 2>&1 &

python ./cmd/dnn_configurable_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/f1-driver-top3" --model FTTransformer --channels 512 --num_layers 2 > logs/FTTransformer_f1-top3.log 2>&1 &

python ./cmd/dnn_configurable_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/ratebeer-beer-positive" --model FTTransformer --channels 256 --num_layers 5 > logs/FTTransformer_ratebeer-beer.log 2>&1 &

python ./cmd/dnn_configurable_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/ratebeer-place-positive" --model FTTransformer --channels 256 --num_layers 2 > logs/FTTransformer_ratebeer-place.log 2>&1 &

python ./cmd/dnn_configurable_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/ratebeer-place-positive-dfs-depth3-feat1000-tw-1" --model FTTransformer --channels 128 --num_layers 3 > logs/FTTransformer_ratebeer-place-dfs.log 2>&1 &

python ./cmd/dnn_configurable_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/ratebeer-user-active" --model FTTransformer --channels 128 --num_layers 3 > logs/FTTransformer_ratebeer-user.log 2>&1 &

python ./cmd/dnn_configurable_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/trial-site-success" --model FTTransformer --channels 64 --num_layers 5 > logs/FTTransformer_trial-site.log 2>&1 &

python ./cmd/dnn_configurable_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/trial-study-adverse" --model FTTransformer --channels 128 --num_layers 2 > logs/FTTransformer_trial-adverse.log 2>&1 &

python ./cmd/dnn_configurable_baseline.py --data_dir "/home/lingze/embedding_fusion/data/dfs-flatten-table/trial-study-outcome" --model FTTransformer --channels 64 --num_layers 4 > logs/FTTransformer_trial-outcome.log 2>&1 &

echo "all training task done！"
echo "check logs: ls logs/"
echo "waiting: wait"
