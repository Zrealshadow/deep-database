# TP-BERTa Package

This package provides preprocessing and training functionality for TP-BERTa models.

## Structure

- `preprocess.py`: Converts CSV rows to embedding strings
- `train.py`: Trains prediction head on preprocessed embeddings
- `__init__.py`: Package initialization

## Usage

### 1. Preprocessing: CSV Rows → Embedding Strings

```python
from LTM.preprocess import process_csv_rows_to_embeddings

# CSV rows as semicolon-separated strings
csv_rows = [
    "1.5;category_a;text_value;0",  # features;label
    "2.3;category_b;text_value;1",
    # ...
]

# Process to embedding strings
embeddings = process_csv_rows_to_embeddings(
    csv_rows=csv_rows,
    pretrain_dir="/path/to/pretrained/tpberta",
    delimiter=";",
    output_format="base64",  # or "comma_separated"
)

# embeddings is a list of strings (base64-encoded embeddings)
```

### 2. Integration with generate_table_data.py

Add `--tpberta_format` flag when generating table data:

```bash
export TPBERTA_PRETRAIN_DIR=/path/to/pretrained/tpberta

python cmds/generate_table_data.py \
    --dbname event \
    --task_name user-attendance \
    --table_output_dir ./data \
    --tpberta_format
```

This will:
1. Generate regular TableData format (CSV files, .pt files)
2. Process CSV rows through TP-BERTa encoder
3. Save embeddings to `{output_dir}/tpberta_embeddings/`:
   - `train_embeddings.csv`
   - `val_embeddings.csv`
   - `test_embeddings.csv`

Each embedding CSV has columns: `['embedding', 'label']` where `embedding` is a base64-encoded string.

### 3. Training Prediction Head

```python
from LTM.train import train_prediction_head

results = train_prediction_head(
    embedding_csv_path="./data/tpberta_embeddings/train_embeddings.csv",
    output_dir="./results/prediction_head",
    embedding_dim=768,  # TP-BERTa hidden size
    embedding_format="base64",
    task_type="binclass",  # or "regression"
    hidden_dims=[256, 128],
    dropout=0.2,
    batch_size=64,
    learning_rate=1e-3,
    num_epochs=200,
    early_stop=50,
)
```

Or from command line:

```python
# Create a simple training script
from LTM.train import train_prediction_head
import sys

if __name__ == "__main__":
    train_prediction_head(
        embedding_csv_path=sys.argv[1],
        output_dir=sys.argv[2],
        embedding_dim=int(sys.argv[3]),
        task_type=sys.argv[4] if len(sys.argv) > 4 else "binclass",
    )
```

## Data Format

### Input CSV Format (for preprocessing)

Each row is a semicolon-separated string:
```
feature1;feature2;feature3;...;label
```

Example:
```
1.5;category_a;some text;0
2.3;category_b;other text;1
```

### Output Embedding CSV Format

After preprocessing, embeddings are saved as CSV with columns:
- `embedding`: Base64-encoded embedding string
- `label`: Target label

## Requirements

- TP-BERTa pretrained model directory (set via `TPBERTA_PRETRAIN_DIR` env var)
- PyTorch
- pandas, numpy
- scikit-learn (for metrics)

## Notes

- Embeddings are extracted from the CLS token (pooled output) of TP-BERTa encoder
- The encoder is frozen during preprocessing (only forward pass)
- Prediction head is a simple MLP that can be trained separately

