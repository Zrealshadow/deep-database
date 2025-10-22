#!/usr/bin/env python3
"""
Re-score Existing Architectures with New ExpressFlow Version

This script:
1. Reads all existing Q-Zero Filter results
2. Extracts unique (dataset, test_group, space, arch, encoder_channels) combinations
3. Re-scores each architecture using a new expressflow version
4. Saves simplified results to explore_proxy/expressflow_v{version}.csv

Output columns:
- dataset
- test_group
- space_name
- architecture
- encoder_channels
- num_blocks
- new_proxy_score (from new expressflow version)
- actual_performance (AUC or MAE from original results)
- metric_type (auc or mae)
"""

import argparse
import glob
import os
import json
import pandas as pd
import torch
from typing import Tuple, Dict

from utils.data import TableData
from model.base import construct_stype_encoder_dict, default_stype_encoder_cls_kwargs
from q_zero.search_space import QZeroMLP, QZeroResNet


def load_existing_results(results_dir: str) -> pd.DataFrame:
    """
    Load all existing Q-Zero Filter results and extract unique architectures
    
    Returns:
        DataFrame with columns: dataset, test_group, space_name, architecture, 
                               encoder_channels, actual_performance, metric_type
    """
    # Find all CSV files
    csv_files = glob.glob(os.path.join(results_dir, '*_n*.csv'))

    if len(csv_files) == 0:
        raise ValueError(f"No CSV files found in {results_dir}")

    print(f"📥 Loading existing results from {len(csv_files)} CSV files...")

    # Load and combine all CSVs
    all_dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            all_dfs.append(df)
        except Exception as e:
            print(f"  ⚠️  Failed to load {csv_file}: {e}")

    if len(all_dfs) == 0:
        raise ValueError("No valid CSV files loaded")

    combined_df = pd.concat(all_dfs, ignore_index=True)

    print(f"  ✅ Loaded {len(combined_df)} total rows")

    # Determine metric type for each row
    combined_df['metric_type'] = combined_df.apply(
        lambda row: 'auc' if 'auc' in combined_df.columns and pd.notna(row.get('auc', None)) else 'mae',
        axis=1
    )

    # Extract unique architectures (deduplicate across selection methods)
    unique_cols = ['dataset', 'space_name', 'test_group', 'architecture', 'encoder_channels', 'num_blocks']

    # Get unique architectures with their performance
    unique_archs = combined_df.drop_duplicates(subset=unique_cols).copy()

    # Add actual performance column
    unique_archs['actual_performance'] = unique_archs.apply(
        lambda row: row['auc'] if row['metric_type'] == 'auc' else row['mae'],
        axis=1
    )

    # Select relevant columns
    result_df = unique_archs[['dataset', 'test_group', 'space_name', 'architecture',
                              'encoder_channels', 'num_blocks', 'actual_performance', 'metric_type']].copy()

    print(f"  ✅ Extracted {len(result_df)} unique architectures")
    print(f"\n  Breakdown:")
    print(f"    • Datasets: {result_df['dataset'].nunique()}")
    print(f"    • MLP architectures: {len(result_df[result_df['space_name'] == 'mlp'])}")
    print(f"    • ResNet architectures: {len(result_df[result_df['space_name'] == 'resnet'])}")

    return result_df


def rescore_architecture_cached(
        dataset: str,
        test_group: int,
        space_name: str,
        architecture: str,
        encoder_channels: int,
        num_blocks: int,
        config: Dict,
        device: torch.device,
        expressflow_version: str,
        stype_encoder_dict: Dict,
        dataset_cache: Dict,
) -> Tuple[float, float]:
    """
    Re-score a single architecture using new expressflow version (with dataset caching)
    
    This function uses THE SAME scoring logic as q_zero_filter.py:
    1. Load dataset (cached) and encode features using a temporary model
    2. Create the target architecture
    3. Call express_flow_score on the architecture's mlp/backbone
    
    Args:
        dataset: Dataset name
        test_group: Test group/timestamp
        space_name: 'mlp' or 'resnet'
        architecture: e.g., "64-128-256"
        encoder_channels: Input dimension for the encoder (from original results)
        num_blocks: Number of blocks/layers
        data_dir_base: Base directory for data
        device: PyTorch device
        expressflow_version: Version identifier for the scoring function
        stype_encoder_dict: Encoder dictionary for torch_frame
        dataset_cache: Cache dictionary to avoid reloading same dataset
    
    Returns:
        (new_proxy_score, elapsed_time)
    """
    # Parse architecture
    arch_dims = [int(x) for x in architecture.split('-')]

    # Load dataset (with caching)
    if dataset not in dataset_cache:
        # Get data_dir from config (same as q_zero_filter.py)
        if dataset not in config:
            print(f"    ⚠️  Dataset {dataset} not found in config")
            return float('nan'), 0.0
        
        data_dir = config[dataset]['data_dir']

        # if "lingze" in data_dir:
        #     data_dir = data_dir.replace("/home/lingze/embedding_fusion/", "./")

        try:
            table_data = TableData.load_from_dir(data_dir)
            dataset_cache[dataset] = table_data
            print(f"    📥 Loaded dataset: {dataset} from {data_dir}")
        except Exception as e:
            print(f"    ⚠️  Failed to load dataset {dataset}: {e}")
            return float('nan'), 0.0
    else:
        table_data = dataset_cache[dataset]

    # Get train_tf and col_stats (same as q_zero_filter.py)
    train_tf = table_data.train_tf
    train_df = table_data.train_df
    col_stats = table_data.col_stats  # ← From table_data, not train_tf
    num_cols = sum(len(v) for v in table_data.col_names_dict.values())
    out_channels = 1
    
    # Get groups/timestamps for this dataset from config
    dataset_config = config[dataset]
    groups = dataset_config['groups']

    # Cache key for encoded batch (includes encoder_channels since it determines encoding)
    # Key insight: Same (dataset, space_name) has same encoder_channels for ALL test_groups!
    # So we can cache at dataset level, not test_group level
    encoded_cache_key = f"{dataset}_{space_name}_{encoder_channels}"

    # Check if we already have encoded features for this config
    if encoded_cache_key not in dataset_cache:
        print(f"..Creating new encoded_cache_key {encoded_cache_key}")
        # Get raw batch first
        batch_cache_key = f"{dataset}_{test_group}_raw"
        
        if batch_cache_key not in dataset_cache:
            print(f"..Creating new batch_cache_key {encoded_cache_key}, batch_cache_key {batch_cache_key}")
            prev_group = test_group - 1
            if prev_group < 0 or prev_group >= len(groups):
                return float('nan'), 0.0

            # Get indices for previous group using timestamp
            prev_dates = groups[prev_group]
            if isinstance(prev_dates, str):
                prev_dates = [prev_dates]
            
            prev_mask = train_df['timestamp'].isin(prev_dates)
            prev_indices = train_df[prev_mask].index.tolist()

            if len(prev_indices) == 0:
                return float('nan'), 0.0

            # Sample a small batch (8 samples) for scoring
            sample_size = min(8, len(prev_indices))
            sample_indices = prev_indices[:sample_size]
            
            # Get raw batch
            sample_batch = train_tf[sample_indices].to(device)
            dataset_cache[batch_cache_key] = sample_batch
        else:
            sample_batch = dataset_cache[batch_cache_key]
        
        # === Encode features (cache by encoder_channels) ===
        # Create temporary model to encode features
        if space_name == 'mlp':
            temp_model = QZeroMLP(
                channels=encoder_channels,
                out_channels=out_channels,
                num_layers=2,
                col_stats=col_stats,
                col_names_dict=table_data.col_names_dict,
                stype_encoder_dict=stype_encoder_dict,
                hidden_dims=[encoder_channels],
                normalization='layer_norm',
                dropout_prob=0.2,
            ).to(device)
        else:  # resnet
            temp_model = QZeroResNet(
                channels=encoder_channels,
                out_channels=out_channels,
                num_layers=2,
                col_stats=col_stats,
                col_names_dict=table_data.col_names_dict,
                stype_encoder_dict=stype_encoder_dict,
                block_widths=[encoder_channels, encoder_channels],
                normalization='layer_norm',
                dropout_prob=0.2,
            ).to(device)

        # Encode features
        with torch.no_grad():
            x_encoded, _ = temp_model.encoder(sample_batch)
            if space_name == 'mlp':
                x_encoded = torch.mean(x_encoded, dim=1)  # [B, encoder_channels]
            else:
                x_encoded = x_encoded.view(x_encoded.size(0), -1)  # [B, encoder_channels * num_cols]

        # Cache encoded features
        dataset_cache[encoded_cache_key] = x_encoded

        del temp_model
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()
    else:
        # Reuse cached encoded features
        x_encoded = dataset_cache[encoded_cache_key]

    sample_batch_x = x_encoded

    # === CREATE TARGET ARCHITECTURE (same as q_zero_filter.py) ===
    num_layers = len(arch_dims) + 1

    if space_name == 'mlp':
        model = QZeroMLP(
            channels=sample_batch_x.shape[1],
            out_channels=out_channels,
            num_layers=num_layers,
            col_stats=col_stats,
            col_names_dict=table_data.col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
            hidden_dims=arch_dims,
            normalization='layer_norm',
            dropout_prob=0.2,
        ).to(device)
        net_for_proxy = model.mlp
    else:  # resnet
        pre_backbone_dim = sample_batch_x.shape[1]
        channels = pre_backbone_dim // num_cols

        model = QZeroResNet(
            channels=channels,
            out_channels=out_channels,
            num_layers=len(arch_dims),
            col_stats=col_stats,
            col_names_dict=table_data.col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
            block_widths=arch_dims,
            normalization='layer_norm',
            dropout_prob=0.2,
        ).to(device)
        net_for_proxy = model.backbone

    if expressflow_version == "v1":
        from q_zero.proxies.expressflow_v1 import express_flow_score
    elif expressflow_version == "v2":
        from q_zero.proxies.expressflow_v2 import express_flow_score
    elif expressflow_version == "v3":
        from q_zero.proxies.expressflow_v3 import express_flow_score
    else:
        raise ValueError(f"Unknown expressflow_version: {expressflow_version}")
    
    try:
        score, elapsed = express_flow_score(
            arch=net_for_proxy,
            batch_data=sample_batch_x,
            device=str(device),  # ← Convert torch.device to string
            use_wo_embedding=False,
            linearize_target=None,
            epsilon=1e-5,
            weight_mode="traj_width",
            use_fp64=False,
        )
    except Exception as e:
        print(f"    ⚠️  Scoring failed for {architecture}: {e}")
        score = float('nan')
        elapsed = 0.0

    # Clean up
    del model
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()

    return float(score), elapsed


def main():
    parser = argparse.ArgumentParser(description='Re-score Architectures with New ExpressFlow Version')

    parser.add_argument(
        '--expressflow_version',
        type=str,
        default='v1',
        help='Version identifier for the expressflow scoring function'
    )
    
    # those can be default
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./result_raw_from_server/q_zero_filter',
        help='Directory containing existing Q-Zero Filter results'
    )
    parser.add_argument(
        '--config_file',
        type=str,
        default='./q_zero_config.json',
        help='Config file path (contains data_dir for each dataset)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./result_raw_from_server/explore_proxy',
        help='Output directory for rescored results'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for scoring'
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 80)
    print("🔄 Re-scoring Architectures with New ExpressFlow Version")
    print("=" * 80)
    print(f"\n📂 Results directory: {args.results_dir}")
    print(f"📂 Config file: {args.config_file}")
    print(f"📂 Output directory: {args.output_dir}")
    print(f"🏷️  ExpressFlow version: {args.expressflow_version}")
    print(f"🖥️  Device: {device}")
    
    # Load config file (same as q_zero_filter.py)
    print(f"\n{'=' * 80}")
    print("📋 Loading Config")
    print(f"{'=' * 80}")
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    print(f"  ✅ Loaded config with {len(config)} datasets")

    # Load existing results
    print(f"\n{'=' * 80}")
    print("📥 Loading Existing Results")
    print(f"{'=' * 80}")

    arch_df = load_existing_results(args.results_dir)

    # Note: stype_encoder_dict is created fresh for each dataset to avoid state issues

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare output
    output_file = os.path.join(args.output_dir, f'expressflow_{args.expressflow_version}.csv')

    print(f"\n{'=' * 80}")
    print(f"🔄 Re-scoring {len(arch_df)} Architectures")
    print(f"{'=' * 80}")

    # Sort by dataset to group architectures from same dataset together
    arch_df_sorted = arch_df.sort_values(['dataset', 'test_group', 'space_name']).reset_index(drop=True)

    results = []
    total = len(arch_df_sorted)

    # Cache for current dataset only (to save memory)
    dataset_cache = {}
    current_dataset = None
    stype_encoder_dict = None

    for idx, row in arch_df_sorted.iterrows():
        dataset_name = row['dataset']

        # Clear cache and recreate encoder when switching to new dataset
        if current_dataset != dataset_name:
            if current_dataset is not None:
                print(f"    🧹 Clearing cache for {current_dataset}")
            
            # Clear ALL cache (including batch cache from previous dataset)
            dataset_cache.clear()
            if str(device).startswith('cuda'):
                torch.cuda.empty_cache()
            
            print(f"\n{'─' * 80}")
            print(f"📊 Processing dataset: {dataset_name}")
            print(f"{'─' * 80}")
            current_dataset = dataset_name
            
            # Create fresh stype_encoder_dict for each dataset (critical to avoid state issues)
            print(f"  🔧 Creating fresh encoder dict for {dataset_name}")
            stype_encoder_dict = construct_stype_encoder_dict(
                default_stype_encoder_cls_kwargs,
            )

        if (idx + 1) % 50 == 0:
            print(f"  Progress: {idx + 1}/{total} ({100 * (idx + 1) / total:.1f}%)")

        # Re-score this architecture (dataset caching happens inside rescore_architecture_cached)
        new_score, elapsed = rescore_architecture_cached(
            dataset=dataset_name,
            test_group=row['test_group'],
            space_name=row['space_name'],
            architecture=row['architecture'],
            encoder_channels=int(row['encoder_channels']),
            num_blocks=int(row['num_blocks']),
            config=config,
            device=device,
            expressflow_version=args.expressflow_version,
            stype_encoder_dict=stype_encoder_dict,
            dataset_cache=dataset_cache,
        )

        # Store result (simplified format, similar to q_zero_filter.py)
        result_row = {
            'dataset': row['dataset'],
            'space_name': row['space_name'],
            'test_group': int(row['test_group']),
            'num_blocks': int(row['num_blocks']),
            'encoder_channels': int(row['encoder_channels']),
            'architecture': row['architecture'],
            'new_proxy_score': new_score,
        }

        # Add actual performance with proper column name (auc or mae)
        if row['metric_type'] == 'auc':
            result_row['auc'] = row['actual_performance']
        else:
            result_row['mae'] = row['actual_performance']

        results.append(result_row)

        # Save incrementally every 100 architectures
        if (idx + 1) % 100 == 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_csv(output_file, index=False)
            print(f"    💾 Saved checkpoint to {output_file}")

    # Final save
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False)

    print(f"\n{'=' * 80}")
    print(f"✅ Re-scoring Complete!")
    print(f"{'=' * 80}")
    print(f"📁 Results saved to: {output_file}")
    print(f"📊 Total architectures rescored: {len(result_df)}")
    print(f"📊 Datasets: {result_df['dataset'].nunique()}")
    print(f"📊 Spaces: {result_df['space_name'].unique().tolist()}")
    print(f"📊 Test groups: {result_df['test_group'].nunique()}")

    # Summary statistics
    print(f"\n📈 Score Summary:")
    print(
        f"  • New proxy scores range: [{result_df['new_proxy_score'].min():.2f}, {result_df['new_proxy_score'].max():.2f}]")
    print(f"  • NaN scores: {result_df['new_proxy_score'].isna().sum()}/{len(result_df)}")
    print(f"  • Average scoring time: {result_df['scoring_time'].mean():.3f}s")


if __name__ == "__main__":
    main()
