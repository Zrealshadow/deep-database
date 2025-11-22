import numpy as np
import os
from pathlib import Path
from tpberta.preprocess import _get_tpberta_embeddings
from utils.data import DatabaseFactory


tasks = [
    ("hm", "user-churn", "rel-hm"),
    ("event", "user-repeat", "rel-event"),
    ("trial", "study-outcome", "rel-trial"),
    ("ratebeer", "user-active", "ratebeer"),
    ("avito", "user-clicks", "rel-avito"),

    ("hm", "item-sales", "rel-hm"),
    ("event", "user-attendance", "rel-event"),
    ("trial", "site-success", "rel-trial"),
    ("ratebeer", "beer-positive", "ratebeer"),
    ("avito", "ad-ctr", "rel-avito")
]
pretrain_dir = "/home/naili/tp-berta/checkpoints/tp-joint"
cache_dir_root = "/home/lingze/.cache/relbench/"

for ele in tasks:
    db_name = ele[0]
    task_name = ele[1]
    cache_dir_db_name = ele[2]
    cache_dir = cache_dir_root + cache_dir_db_name
    print(f"cache_dir={cache_dir} for {db_name}")

    db = DatabaseFactory.get_db(db_name,
                                cache_dir=cache_dir,
                                upto_test_timestamp=False,
                                with_text_compress=True)
    dataset = DatabaseFactory.get_dataset(db_name, cache_dir)
    task = DatabaseFactory.get_task(db_name, task_name, dataset)
    entity_table = task.entity_table
    used_df = db.table_dict[entity_table]

    embeddings = _get_tpberta_embeddings(
        df=used_df,
        pretrain_dir=pretrain_dir,
        has_label=False,  # No label column
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
    )

    print(f"\nOutput embeddings shape: {embeddings.shape}")
    print(f"Output embeddings dtype: {embeddings.dtype}")

    # Save to numpy array file: {db_name}_{task_name}_data.npy
    output_dir = Path("/home/naili/sharing-embedding-table/data/tpberta_relbench")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = f"{db_name}_{task_name}_data.npy"
    output_path = output_dir / output_filename
    np.save(output_path, embeddings)
    print(f"Embeddings saved to: {output_path}")

