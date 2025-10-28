import argparse

from utils.logger import ModernLogger
from utils.data import TableData

from torch_frame import stype
from typing import Dict


def describe_table_data(table_data:TableData)-> Dict:
    desc = {}
    desc["num_train_rows"] = len(table_data.train_df)
    desc["num_val_rows"] = len(table_data.val_df)
    desc["num_test_rows"] = len(table_data.test_df)
    desc["num_columns"] = len(table_data.col_to_stype)
    num_numeric_cols = sum(
        1 for col_type in table_data.col_to_stype.values() if col_type == stype.numerical)
    num_categorical_cols = sum(
        1 for col_type in table_data.col_to_stype.values() if col_type == stype.categorical)
    num_text_cols = sum(
        1 for col_type in table_data.col_to_stype.values() if col_type == stype.text_embedded)
    num_embedding_cols = sum(
        1 for col_type in table_data.col_to_stype.values() if col_type == stype.embedding)

    desc["num_numeric_cols"] = num_numeric_cols
    desc["num_categorical_cols"] = num_categorical_cols
    desc["num_text_cols"] = num_text_cols
    desc["num_embedding_cols"] = num_embedding_cols
    return desc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Tabular data statics")

    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the tabular data directory.")

    args = parser.parse_args()

    table_data = TableData.load_from_dir(args.data_dir)

    desc = describe_table_data(table_data)

    logger = ModernLogger(
        name="Tabular_Statics",
        level="info"
    )

    logger.info(f"Describe of the tabular data in {args.data_dir}")
    task_info = ""
    for key, value in desc.items():
        task_info += f"{key}: {value}\n"

    logger.info_panel("Tabular Data Statics", task_info)
    
