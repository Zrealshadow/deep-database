# Retrieval Indices Cache Directory

This directory stores BM25 retrieval indices for each entity table in the datasets.

## Purpose

RAM uses BM25-based information retrieval to discover semantic relationships between entities. Pre-built indices enable fast retrieval during both preprocessing (for edge discovery) and training (for dynamic retrieval augmentation).

## Contents

Each subdirectory corresponds to a specific dataset, containing BM25 retriever models:

- **Entity-level retrievers**: One retriever per entity table (e.g., `Users_retriever_bm25`, `Items_retriever_bm25`)
- **Index files**: Serialized BM25 indices with term statistics
- **Numba-optimized scorers**: Compiled scoring functions for fast retrieval

## Directory Structure

```
tmp/
├── rel-avito/
│   ├── Ads_retriever_bm25/
│   ├── Users_retriever_bm25/
│   └── ...
├── rel-stack/
│   ├── users_retriever_bm25/
│   ├── posts_retriever_bm25/
│   └── ...
└── ...
```

## Generation Process

Retrieval indices are built during preprocessing:

1. **Document Creation**: Generate text documents for each entity via random walks
2. **Index Building**: Create BM25 index using `bm25s` library
3. **Numba Compilation**: Activate optimized scoring for fast retrieval
4. **Serialization**: Save indices to disk for later loading

Example from preprocessing:

```python
import bm25s

# Build retriever
retriever = bm25s.BM25(backend="numba")
retriever.index(documents)
retriever.activate_numba_scorer()

# Save to disk
retriever.save(f"./ram/tmp/rel-avito/Ads_retriever_bm25")
```

## Loading Indices

During training or inference, load pre-built indices:

```python
import bm25s

retriever = bm25s.BM25.load("./ram/tmp/rel-avito/Ads_retriever_bm25")
retriever.activate_numba_scorer()

# Use for retrieval
related_docs, scores = retriever.retrieve(queries, k=20, n_threads=-1)
```

## Performance Considerations

- **Storage**: Indices can be large (hundreds of MB to several GB per dataset)
- **Memory**: Loading all indices for a large database requires significant RAM
- **Speed**: Numba backend provides 10-100x speedup over pure Python
- **Parallelization**: Multi-threaded retrieval supported via `n_threads` parameter

## Maintenance

- Indices must be regenerated if:
  - Document generation parameters change (walk length, round, etc.)
  - Tokenization strategy is modified
  - Dataset is updated
- Can be safely deleted and regenerated using preprocessing scripts
- Consider using `.gitignore` to exclude from version control due to size
