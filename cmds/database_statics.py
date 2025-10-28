import argparse


from utils.data import DatabaseFactory
from relbench.base import Database
from utils.logger import ModernLogger

from typing import Dict


def describe_database(db: Database) -> Dict:

    num_tables = len(db.table_dict)
    num_columns = 0
    num_relationships = 0
    for _, table in db.table_dict.items():
        num_columns += len(table.df.columns)
        num_relationships += len(table.fkey_col_to_pkey_table)
    desc = {
        "num_tables": num_tables,
        "num_columns": num_columns,
        "num_relationships": num_relationships
    }
    return desc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Database static information extractor")

    parser.add_argument("--db_name", type=str, required=True,
                        help="Name of the database.")

    args = parser.parse_args()

    db = DatabaseFactory.get_db(args.db_name)

    desc = describe_database(db)

    logger = ModernLogger(
        name="Database_Statics",
        level="info"
    )
    logger.info(f"Describe of the database: {args.db_name}")
    db_info = ""
    for key, value in desc.items():
        db_info += f"{key}: {value}\n"
    logger.info_panel("Database Statics", db_info)
