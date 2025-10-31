# Inter-Table Cross-Edges Directory

This directory stores retrieval-augmented cross-table edges discovered through semantic similarity.

## Purpose

RAM augments the explicit database schema with implicit semantic connections discovered via retrieval. These cross-table edges connect entities from different tables based on content similarity, enriching the graph structure beyond foreign key relationships.

## Contents

Each `.npz` file corresponds to a specific dataset and contains cross-table edge arrays:

```
edges/
├── rel-avito-edges.npz
├── rel-avito-Ads-edges.npz
├── rel-stack-edges.npz
└── ...
```

## Data Format

Each `.npz` file contains multiple edge arrays, one per table pair:

```python
import numpy as np

# Load edges
data = np.load("ram/edges/rel-avito-Ads-edges.npz")

# Keys are formatted as "{source_table}-{destination_table}"
for edge_key in data.keys():
    edge_array = data[edge_key]  # Shape: [num_edges, 2]
    src_table, dst_table = edge_key.split('-')
    print(f"{edge_key}: {edge_array.shape[0]} edges")
```

### Array Structure

- **Shape**: `[num_edges, 2]`
- **Column 0**: Source entity primary key indices
- **Column 1**: Destination entity primary key indices
- **Direction**: Edges are directional (source queries destination)

## Generation Process

Cross-table edges are discovered during preprocessing:

1. **Candidate Selection**: Identify table pairs with no direct foreign key relationship
2. **Multi-hop Check**: Verify tables are not 1-hop neighbors in schema graph
3. **Document Sampling**: Sample entities from source table (50% or min 4096)
4. **Cross-Retrieval**: Query destination table's retriever with source documents
5. **Score Filtering**: Keep edges with score > mean + 2×std
6. **Edge Construction**: Pair filtered results with corresponding query entities

Example filtering logic:

```python
# Retrieve top-k most similar entities
related_pkys, scores = retriever.retrieve(source_docs, k=20)

# Statistical threshold
threshold = scores.mean() + 2 * scores.std()
mask = scores > threshold

# Build edges
src_indices = np.repeat(source_pkys, mask.sum(axis=1))
dst_indices = related_pkys[mask]
edges = np.stack([src_indices, dst_indices], axis=1)
```

## Edge Statistics

Typical characteristics:

- **Density**: 100-10,000+ edges per table pair
- **Selectivity**: ~1-5% of possible pairs pass the threshold
- **Asymmetry**: Edge counts may differ for (A→B) vs (B→A)
- **Quality**: Higher scores indicate stronger semantic similarity

## Usage in Model

Cross-table edges augment the graph structure:

1. **Schema Graph**: Start with explicit foreign key relationships
2. **Semantic Graph**: Add retrieval-discovered cross-table edges
3. **Unified Graph**: Merge into single heterogeneous graph
4. **Message Passing**: Enable information flow across semantically related entities

## Graph Augmentation Strategy

```python
# Load explicit edges from schema
schema_edges = load_schema_edges(database)

# Load retrieval edges
retrieval_edges = np.load("ram/edges/rel-avito-Ads-edges.npz")

# Combine for heterogeneous graph
for edge_type, edge_array in retrieval_edges.items():
    src_table, dst_table = edge_type.split('-')
    # Add as new edge type in PyG HeteroData
    hetero_data[src_table, f'semantic_to', dst_table].edge_index = edge_array.T
```

## Maintenance

Regenerate edges if:

- Retrieval indices change
- Similarity threshold is modified
- Sampling parameters change (sample_size, topn)
- Dataset is updated

Delete and regenerate using preprocessing scripts (`ram/preprocess/*.py`).

## Performance Impact

- **Graph Size**: Increases total edges by 20-100%
- **Connectivity**: Reduces average shortest path length
- **Semantics**: Captures implicit relationships beyond schema
- **Training**: May increase memory usage but improves model expressiveness
