#!/usr/bin/env python3
"""
Retry Failed Tasks Script
Processes specific failed tasks with specified model and batch_size
"""

import numpy as np
import os
import argparse
from pathlib import Path
from LTM import get_embeddings
from utils.data import DatabaseFactory


def process_task(db_name, task_name, cache_dir_db_name, model, batch_size=32):
    """Process a single task with specified model."""
    cache_dir_root = "/home/lingze/.cache/relbench/"
    cache_dir = cache_dir_root + cache_dir_db_name
    pretrain_dir = "/home/naili/tp-berta/checkpoints/tp-joint"
    
    print(f"Processing {db_name}:{task_name} with {model}")
    
    try:
        db = DatabaseFactory.get_db(
            db_name,
            cache_dir=cache_dir,
            upto_test_timestamp=False,
            with_text_compress=True
        )
        dataset = DatabaseFactory.get_dataset(db_name, cache_dir)
        task = DatabaseFactory.get_task(db_name, task_name, dataset)
        entity_table = task.entity_table
        used_df = db.table_dict[entity_table].df
        
        print(f"  DataFrame shape: {used_df.shape}")
        
        # Get embeddings
        embeddings = get_embeddings(
            df=used_df,
            model=model,
            dataset_name=f"{db_name}_{task_name}",
            pretrain_dir=pretrain_dir if model == "tpberta" else None,
            has_label=False,
            device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
            batch_size=batch_size,
        )
        
        print(f"  Embeddings shape: {embeddings.shape}")
        
        # Save embeddings
        output_dir = Path("/data/naili/tpberta_relbench") / model
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = f"{db_name}_{task_name}_data.npy"
        output_path = output_dir / output_filename
        np.save(output_path, embeddings)
        print(f"  ✅ Saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Retry failed tasks")
    parser.add_argument("--db_name", type=str, required=True, help="Database name")
    parser.add_argument("--task_name", type=str, required=True, help="Task name")
    parser.add_argument("--cache_dir", type=str, required=True, help="Cache directory name")
    parser.add_argument("--model", type=str, required=True, choices=["tpberta", "nomic", "bge"], help="Model name")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    
    args = parser.parse_args()
    
    success = process_task(
        args.db_name,
        args.task_name,
        args.cache_dir,
        args.model,
        args.batch_size
    )
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main()

