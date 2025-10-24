# Leva: Random Walk + Word2Vec for Table Augmentation. 

Implementation of the approach from **"Leva: Boosting Machine Learning Performance with Relational Embedding Data Augmentation"**.

Simple baseline using **Random Walk + Word2Vec → MLP** for relational tabular prediction.

## Workflow

### 1. Graph Construction
Build a homogeneous graph from relational database:
- **Nodes**: Represent table rows (entities)
- **Edges**: Derived from foreign key relationships

### 2. Embedding Generation
Learn node representations via Random Walk + Word2Vec:
- **Random Walk**: Generate node sequences by randomly traversing edges
- **Word2Vec**: Train skip-gram model on walk sequences to learn embeddings
- **Output**: Dense vector representation for each node/row

### 3. Feature Augmentation
Combine learned embeddings with original tabular features:
- **Lookup**: Map target table rows to their node embeddings
- **Concatenate**: `[original_features | node_embedding]`

### 4. Prediction
Train MLP on augmented features:
- **Classification**: Binary/multi-class with cross-entropy loss
- **Regression**: Continuous prediction with MAE/MSE loss

## Usage

```bash
python train.py \
  --db_name event \
  --task_name user-attendance \
  --embedding_dim 128 \
  --walk_length 20 \
  --walks_per_node 10
```

**Key Parameters:**
- `embedding_dim`: Embedding dimension (default: 128)
- `walk_length`: Steps per random walk (default: 20)
- `walks_per_node`: Number of walks starting from each node (default: 10)

## Results

Augmenting tabular features with graph embeddings improves prediction performance on relational benchmarks (Event, Stack, Trial, Avito).

**Metrics**: ROC-AUC (classification), MAE (regression)

## Reference

> **Leva: Boosting Machine Learning Performance with Relational Embedding Data Augmentation**
