export PYTHONPATH=$(pwd)

# pre-train 
python ./cmd/pretrain.py \
    --tf_cache_dir ./data/rel-event-tensor-frame \
    --db_name event \
    --edge_path ./edges/rel-event-edges.npz \
    --sample_path ./samples/rel-event-samples.npz 



# Event

# with all
python -m cmd.run \
    --tf_cache_dir ./data/rel-event-tensor-frame \
    --db_name event \
    --task_name user-attendance \
    --pretrain_path ./static/method/rel-event-pre-trained.pth \
    --edge_path ./edges/rel-event-edges.npz 

# only - Pre-trained
python -m cmd.run \
    --tf_cache_dir ./data/rel-event-tensor-frame \
    --db_name event \
    --task_name user-attendance \
    --pretrain_path ./static/method/rel-event-pre-trained-wo-edges.pth 


# only - Edges
python -m cmd.run \
    --tf_cache_dir ./data/rel-event-tensor-frame \
    --db_name event \
    --task_name user-attendance \
    --edge_path ./edges/rel-event-edges.npz 





# trial


# with all
python -m cmd.run \
    --tf_cache_dir ./data/rel-trial-tensor-frame \
    --db_name trial \
    --task_name study-adverse \
    --pretrain_path ./static/method/rel-trial-pre-trained.pth \
    --edge_path ./edges/rel-trial-edges.npz \
        --lr 0.01 \
    --max_round_epoch 30

# only - Pre-trained
python -m cmd.run \
    --tf_cache_dir ./data/rel-trial-tensor-frame \
    --db_name trial \
    --task_name study-adverse \
    --pretrain_path ./static/method/rel-trial-pre-trained-wo-edges.pth \
        --lr 0.01 \
    --max_round_epoch 30
    

# only - Edges
python -m cmd.run \
    --tf_cache_dir ./data/rel-trial-tensor-frame \
    --db_name trial \
    --task_name study-adverse \
    --edge_path ./edges/rel-trial-edges.npz \
    --lr 0.01 \
    --max_round_epoch 30



python -m cmd.run \
    --tf_cache_dir ./data/ratebeer-tensor-frame \
    --data_cache_dir /home/lingze/.cache/relbench/ratebeer \
    --db_name ratebeer \
    --task_name place-positive \
    --pretrain_path ./static/method/ratebeer-pre-trained.pth \
    --edge_path ./edges/ratebeer-edges.npz 