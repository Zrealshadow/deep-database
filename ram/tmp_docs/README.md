# Tokenized Documents Directory

This directory stores tokenized document representations of relational database tables.

## Purpose

During the RAM preprocessing pipeline, relational data is converted to text documents for retrieval-based operations. These tokenized documents serve as intermediate representations before building BM25 retrieval indices.

## Contents

Each subdirectory corresponds to a specific dataset (e.g., `rel-avito`, `rel-stack`, etc.) and contains:

- **Tokenized columns**: Text representations of table columns based on inferred types
- **Document corpus**: Collections of documents generated via random walks on the homogeneous graph
- **Metadata**: Information about tokenization strategies and column types

## Generation Process

1. **Column Type Inference**: Automatically detect column types (categorical, numerical, text, etc.)
2. **Tokenization**: Apply type-specific tokenization strategies
   - Categorical: Direct string representation
   - Numerical: Formatted string with proper precision
   - Text: Cleaned and normalized text
   - Temporal: Human-readable date/time formats
3. **Document Generation**: Create documents via random walks connecting related entities

## Usage

These documents are consumed by the retrieval index building process in the next stage of preprocessing. They are not used directly during model training.

## Directory Structure

```
tmp_docs/
├── rel-avito/
│   ├── tokenized_columns/
│   └── documents/
├── rel-stack/
│   ├── tokenized_columns/
│   └── documents/
└── ...
```

## File Formats

- Typically stored as pickled Python objects or JSON files
- Contains mappings from entity IDs to text representations
- Optimized for efficient loading during retrieval index construction

## Notes

- This directory can become large for datasets with many entities
- Documents are regenerated if tokenization parameters change
- Can be safely deleted and regenerated using preprocessing scripts
