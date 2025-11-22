import os
from tpberta.preprocess import process_csv_rows_to_embeddings

"conda activate /home/naili/miniconda3/envs/deepdb"

pretrain_dir = "/home/naili/tp-berta/checkpoints/tp-joint"

test_csv_rows = [
    "25;male;engineer;50000;1",
    "30;female;teacher;45000;0",
    "35;male;doctor;80000;1",
]

embedding_strings = process_csv_rows_to_embeddings(
    csv_rows=test_csv_rows,
    pretrain_dir=pretrain_dir,
    delimiter=";",
    device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
)


