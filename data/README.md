# Data Directory

This directory stores generated tabular datasets and materialized database graph representations (TensorFrames) for various relational databases and prediction tasks.

## Overview

The data directory contains two main types of artifacts:

1. **Tabular Data**: Flattened relational data with various feature engineering levels
2. **TensorFrame Data**: Materialized database graph structures stored as PyTorch tensors


## Contents
mainly includes two type of data:
- `tabular/`: Flattened tabular datasets for each relational database and prediction task
- `tensorframe/`: Materialized database graph representations (TensorFrames) for each relational database


### Use Cases

TensorFrame data is used for:
- **Tabular Models**: Training models on flattened relational data
- **Graph Neural Networks**: Building heterogeneous graphs from database schema
- **Relational Models**: Training models that leverage table relationships
- **Baseline Models**: Graph-based baselines (R-GCN, R-GAT, HGT)

## Generation Scripts


## Notes
- Data files are excluded from git via `.gitignore`, you can download them from official website.
- Reproducibility: All data can be regenerated from raw RelBench datasets
- Storage: Consider using symlinks for large datasets to save space
- Parallel generation: Some scripts support parallel execution for faster generation
