# LTM (Learning Tabular Models)

Unified interface for extracting embeddings and training prediction heads. Supports TP-BERTa, Nomic, and BGE.

## Quick Start

### 1. Generate Embeddings for RelBench (.npy)

```bash
cd LTM/scripts
./save_embed_numpy.sh
```

**Output Structure**:
```
data/tpberta_relbench/
├── nomic/
│   ├── hm_user-churn_data.npy
│   ├── avito_user-clicks_data.npy
│   └── ...
├── bge/
│   ├── hm_user-churn_data.npy
│   └── ...
└── tpberta/
    ├── hm_user-churn_data.npy
    └── ...
```

**Logs**: `logs/run_embeddings_{timestamp}.log`

---

### 2. Preprocess Medium Tables (CSV)

```bash
cd LTM/scripts
./save_medium_embed_csv.sh              # All
./save_medium_embed_csv.sh avito-user-clicks  # Single
```

**Input Structure**:
```
data/fit-medium-table/
├── avito-user-clicks/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── target_col.txt
└── ...
```

**Output Structure**:
```
data/tpberta_table/
├── nomic/
│   ├── avito-user-clicks/
│   │   ├── train.csv          # embedding, target
│   │   ├── val.csv
│   │   ├── test.csv
│   │   └── feature_names.json
│   └── ...
├── bge/
│   └── ...
└── tpberta/
    └── ...
```

**Datasets**: avito-user-clicks, avito-ad-ctr, event-user-repeat, event-user-attendance, ratebeer-beer-positive, ratebeer-place-positive, ratebeer-user-active, trial-site-success, trial-study-outcome, hm-item-sales, hm-user-churn

---

### 3. Train Prediction Head

```bash
cd LTM/scripts
./tpberta_medium_baseline.sh            # All
./tpberta_medium_baseline.sh avito-user-clicks  # Single
```

**Input Structure**:
```
data/tpberta_table/
├── nomic/
│   └── avito-user-clicks/     # From step 2
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
└── ...
```

**Output Structure**:
```
result_raw_from_server/
├── nomic_head/
│   ├── avito-user-clicks/
│   │   ├── results.json       # metrics
│   │   ├── test_predictions.npy
│   │   └── test_targets.npy
│   └── ...
├── bge_head/
│   └── ...
└── tpberta_head/
    └── ...
```

---

## Python API

### Extract Embeddings

```python
from LTM import get_embeddings
import pandas as pd

df = pd.read_csv("data.csv")

# TP-BERTa
emb = get_embeddings(df, model="tpberta", pretrain_dir="...", has_label=False)

# Nomic
emb = get_embeddings(df, model="nomic", task_prefix="classification", batch_size=32)

# BGE
emb = get_embeddings(df, model="bge", batch_size=32)
```

---

## Environment Variables

```bash
export TPBERTA_ROOT="/home/naili/tp-berta"
export TPBERTA_PRETRAIN_DIR="$TPBERTA_ROOT/checkpoints/tp-joint"
export TPBERTA_BASE_MODEL_DIR="$TPBERTA_ROOT/checkpoints/roberta-base"
export PYTHONPATH="$PROJECT_ROOT:$TPBERTA_ROOT:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
```

---

## Models

| Model | Type | Config |
|-------|------|--------|
| **TP-BERTa** | Table transformer | Requires `TPBERTA_PRETRAIN_DIR` |
| **Nomic** | Text embedding | Task prefix: `"classification"`, `"search_document"`, etc. |
| **BGE** | Text embedding | No special config |

---

