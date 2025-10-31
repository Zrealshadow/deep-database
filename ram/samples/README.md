# Intra-Table Positive Samples Directory

This directory stores positive sample pools for contrastive learning in RAM pretraining.

## Purpose

RAM uses contrastive learning to learn entity representations. Positive samples (similar entities within the same table) are discovered via retrieval and cached for efficient training.

## Contents

Each `.npz` file corresponds to a specific dataset and contains positive sample pools for entity tables:

```
samples/
├── rel-avito-samples.npz
├── rel-avito-Ads-samples.npz
├── rel-stack-samples.npz
└── ...
```

## Data Format

Each `.npz` file contains multiple arrays, one per entity table:

```python
import numpy as np

# Load samples
data = np.load("ram/samples/rel-avito-Ads-samples.npz")

# Keys are entity table names
for entity_table in data.keys():
    positive_pool = data[entity_table]  # Shape: [num_entities, max_positives]
    print(f"{entity_table}: {positive_pool.shape}")
```

### Array Structure

- **Rows**: Each row corresponds to one entity (identified by primary key index)
- **Columns**: Each column is a positive sample (similar entity primary key)
- **Values**: Primary key indices of similar entities, or `-1` for padding
- **Selection**: Only entities with at least one positive (beyond self) are included

## Generation Process

Positive samples are discovered during preprocessing:

1. **Self-Retrieval**: Query each entity against its own table's retriever
2. **Similarity Filtering**: Keep results with score > threshold (typically 0.7 × top score)
3. **Self-Exclusion**: Remove the query entity itself (most similar by definition)
4. **Quality Control**: Only include entities with multiple positives
5. **Serialization**: Save as compressed `.npz` arrays

Example threshold calculation:

```python
# Top-k retrieval (k=21 to allow excluding self)
related_pkys, scores = retriever.retrieve(entity_docs, k=21)

# Filter by relative similarity threshold
threshold = 0.7
mask = scores > (scores[:, [0]] * threshold)

# Exclude self (first result) and pad filtered results
related_pkys[~mask] = -1
positive_pool = related_pkys[mask.sum(axis=1) > 1]  # Keep only multi-positive entities
```

## Usage in Training

During contrastive pretraining, positive samples augment the training signal:

1. **Anchor**: Sample an entity from the positive pool
2. **Positive**: Randomly select one of its cached similar entities
3. **Negative**: Sample other entities from the same batch (in-batch negatives)
4. **Objective**: Maximize agreement between anchor and positive, minimize with negatives

## Statistics

Typical positive pool characteristics:

- **Coverage**: 30-70% of entities have discoverable positives
- **Density**: Average 2-10 positives per entity
- **Sparsity**: Padded with `-1` for efficient batching
- **Size**: 10-500 MB per dataset depending on entity count

## Maintenance

Regenerate samples if:

- Retrieval indices are rebuilt
- Similarity threshold changes
- Random walk parameters are modified
- Dataset is updated

Delete and regenerate using preprocessing scripts (`ram/preprocess/*.py`).
