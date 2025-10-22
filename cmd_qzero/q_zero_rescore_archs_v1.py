#!/usr/bin/env python3
"""
Re-score Existing Architectures with New ExpressFlow Version (Fast Version)

Since new expressflow uses torch.ones_like(batch_data) - it doesn't use real data values!
We can skip data loading entirely and just create dummy tensors based on architecture specs.

This script:
1. Reads all existing Q-Zero Filter results
2. Extracts unique (dataset, test_group, space, arch, encoder_channels) combinations
3. Creates dummy input tensors based on encoder_channels
4. Re-scores each architecture using new expressflow version
5. Saves results to explore_proxy/expressflow_v{version}.csv
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
    """
    csv_files = glob.glob(os.path.join(results_dir, '*_n*.csv'))
    
    if len(csv_files) == 0:
        raise ValueError(f"No CSV files found in {results_dir}")
    
    print(f"📥 Loading existing results from {len(csv_files)} CSV files...")
    
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
    
    # Determine metric type
    combined_df['metric_type'] = combined_df.apply(
        lambda row: 'auc' if 'auc' in combined_df.columns and pd.notna(row.get('auc', None)) else 'mae',
        axis=1
    )
    
    # Extract unique architectures
    unique_cols = ['dataset', 'space_name', 'test_group', 'architecture', 'encoder_channels', 'num_blocks']
    unique_archs = combined_df.drop_duplicates(subset=unique_cols).copy()
    
    # Add actual performance
    unique_archs['actual_performance'] = unique_archs.apply(
        lambda row: row['auc'] if row['metric_type'] == 'auc' else row['mae'],
        axis=1
    )
    
    result_df = unique_archs[['dataset', 'test_group', 'space_name', 'architecture',
                              'encoder_channels', 'num_blocks', 'actual_performance', 'metric_type']].copy()
    
    print(f"  ✅ Extracted {len(result_df)} unique architectures")
    print(f"    • MLP: {len(result_df[result_df['space_name'] == 'mlp'])}")
    print(f"    • ResNet: {len(result_df[result_df['space_name'] == 'resnet'])}")
    
    return result_df


def rescore_architecture_fast(
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
    Re-score using QZeroMLP/QZeroResNet (same as q_zero_filter.py)
    
    Load real data and use proper model creation
    """
    # Parse architecture
    arch_dims = [int(x) for x in architecture.split('-')]
    
    # Load dataset (with caching)
    if dataset not in dataset_cache:
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
    
    # Get data info (only need col_stats, col_names_dict, num_cols)
    col_stats = table_data.col_stats
    num_cols = sum(len(v) for v in table_data.col_names_dict.values())
    out_channels = 1
    
    # Create dummy encoded features (all ones)
    # No need for real data or encoding since expressflow uses torch.ones_like anyway!
    batch_size = 8
    
    if space_name == 'mlp':
        # MLP: sample_batch_x shape = [batch_size, encoder_channels]
        sample_batch_x = torch.ones(batch_size, encoder_channels, dtype=torch.float32).to(device)
    else:  # resnet
        # ResNet: sample_batch_x shape = [batch_size, encoder_channels * num_cols]
        sample_batch_x = torch.ones(batch_size, encoder_channels * num_cols, dtype=torch.float32).to(device)
    
    # Create target architecture using QZeroMLP/QZeroResNet
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
    else:
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
    
    # Call expressflow scoring
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
            device=str(device),
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
    parser = argparse.ArgumentParser(description='Re-score Architectures (Fast - No Data Loading)')

    parser.add_argument(
        '--expressflow_version',
        type=str,
        default='v1',
        help='Version identifier for the expressflow scoring function'
    )
    
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
        help='Config file (to get num_cols for each dataset)'
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
    print("🔄 Re-scoring Architectures (Fast Mode - No Data Loading)")
    print("=" * 80)
    print(f"\n📂 Results directory: {args.results_dir}")
    print(f"📂 Config file: {args.config_file}")
    print(f"📂 Output directory: {args.output_dir}")
    print(f"🏷️  ExpressFlow version: {args.expressflow_version}")
    print(f"🖥️  Device: {device}")

    # Load config to get num_cols for each dataset
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
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare output
    output_file = os.path.join(args.output_dir, f'expressflow_{args.expressflow_version}.csv')

    print(f"\n{'=' * 80}")
    print(f"🔄 Re-scoring {len(arch_df)} Architectures (Fast Mode)")
    print(f"{'=' * 80}")
    print("💡 Using dummy tensors (expressflow doesn't use real data values)")

    # Sort by dataset to group architectures
    arch_df_sorted = arch_df.sort_values(['dataset', 'test_group', 'space_name']).reset_index(drop=True)

    results = []
    total = len(arch_df_sorted)
    
    # Cache for current dataset
    dataset_cache = {}
    current_dataset = None
    stype_encoder_dict = None

    for idx, row in arch_df_sorted.iterrows():
        dataset_name = row['dataset']

        # Switch dataset: clear cache and create new encoder
        if current_dataset != dataset_name:
            if current_dataset is not None:
                print(f"    🧹 Clearing cache for {current_dataset}")
            
            dataset_cache.clear()
            if str(device).startswith('cuda'):
                torch.cuda.empty_cache()
            
            print(f"\n{'─' * 80}")
            print(f"📊 Processing dataset: {dataset_name}")
            print(f"{'─' * 80}")
            current_dataset = dataset_name
            
            # Create fresh encoder dict
            print(f"  🔧 Creating fresh encoder dict for {dataset_name}")
            stype_encoder_dict = construct_stype_encoder_dict(
                default_stype_encoder_cls_kwargs,
            )

        if (idx + 1) % 50 == 0:
            print(f"  Progress: {idx + 1}/{total} ({100 * (idx + 1) / total:.1f}%)")

        # Re-score using QZeroMLP/QZeroResNet with dummy tensors
        new_score, elapsed = rescore_architecture_fast(
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

        # Store result
        result_row = {
            'dataset': row['dataset'],
            'space_name': row['space_name'],
            'test_group': int(row['test_group']),
            'num_blocks': int(row['num_blocks']),
            'encoder_channels': int(row['encoder_channels']),
            'architecture': row['architecture'],
            'new_proxy_score': new_score,
        }
        
        # Add actual performance
        if row['metric_type'] == 'auc':
            result_row['auc'] = row['actual_performance']
        else:
            result_row['mae'] = row['actual_performance']
        
        results.append(result_row)
        
        # Save checkpoint every 500 architectures
        if (idx + 1) % 500 == 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_csv(output_file, index=False)
            print(f"    💾 Checkpoint: {idx+1}/{total}")

    # Final save
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False)

    print(f"\n{'=' * 80}")
    print(f"✅ Re-scoring Complete!")
    print(f"{'=' * 80}")
    print(f"📁 Results: {output_file}")
    print(f"📊 Total: {len(result_df)} architectures")
    print(f"📊 Datasets: {result_df['dataset'].nunique()}")
    print(f"📊 NaN scores: {result_df['new_proxy_score'].isna().sum()}/{len(result_df)}")


if __name__ == "__main__":
    main()

