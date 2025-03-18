import math
import os
import numpy as np
import pandas as pd

from relbench.base import Database
from torch_frame import stype
from torch_frame.utils import infer_df_stype
from typing import Dict, Optional, Tuple, List

from utils.util import get_keyword_model
from dataclasses import dataclass
from tqdm import tqdm


def infer_type_in_db(
    db: Database,
    verbose: bool = False,
) -> Dict[str, Dict[str, stype]]:
    r"""Propose stype for columns of a set of tables in the given database.

    Args:
        db (Database): The database object containing a set of tables.
    Returns:
        Dict[str, Dict[str, Any]]: A dictionary mapping table name into
            :obj:`col_to_stype` (mapping column names into inferred stypes).
    """

    inferred_col_to_stype_dict = {}

    for table_name, table in db.table_dict.items():
        df = table.df
        df = df.sample(min(10_000, len(df)))
        inferred_col_to_stype = infer_df_stype(df)

        inferred_col_to_stype_dict[table_name] = basic_infer_stype(
            df,
            inferred_col_to_stype,
            table_name,
            verbose)

    return inferred_col_to_stype_dict


# rule 0
# based on the column name
# we predefined some
numerical_keywords = [
    'count', 'num', 'amount', 'total', 'length', 'height', 'value', 'rate',  'number',
    'score', 'level', 'size', 'price', 'percent', 'ratio', 'volume', 'index', 'avg', 'max', 'min', 'age'
]

categorical_keywords = [
    'type', 'category', 'class', 'label', 'status', 'code', 'id',
    'region', 'zone', 'flag', 'is_', 'has_', 'mode', 'duration'
]

text_keywords = [
    'description', 'comments', 'content', 'name', 'review', 'message', 'note', 'query', 'summary'
]


def basic_infer_stype(
    df: pd.DataFrame,
    col_to_stype: Dict[str, stype],
    prefix: str = "",
    verbose: bool = False,
) -> Dict[str, stype]:
    for col_name in df.columns:
        if col_name not in col_to_stype:
            continue
        guess_type = col_to_stype[col_name]

        guess_type = custom_rule_infer(
            df, col_name, guess_type, prefix, verbose)
        guess_type = custom_rule_1_infer(
            df, col_name, guess_type, prefix, verbose)

        col_to_stype[col_name] = guess_type
    return col_to_stype


def custom_rule_infer(
    df: pd.DataFrame,
    col_name: str,
    guess_type: stype,
    prefix: str = "",
    verbose: bool = False,
) -> stype:
    """rule_0 mainly rule based on the column name
    """
    if any([kw in col_name.lower() for kw in text_keywords]):
        
        if verbose and guess_type != stype.text_embedded:
            print(
                f"[rule 0]: {prefix}Inferred {col_name} from {guess_type} as text_embedded")
        
        return stype.text_embedded

    # rule 0: to numerical stype
    if any([kw in col_name.lower() for kw in numerical_keywords]):
        if guess_type == stype.numerical:
            return guess_type

        # check the data can be converted to numerical
        is_convertible = (
            pd.to_numeric(df[col_name], errors='coerce').notna() +
            df[col_name].isna()
        ).all()

        if not is_convertible:
            return guess_type

        if verbose:
            print(
                f"[rule 0]: {prefix}Inferred {col_name} from {guess_type} as numerical")

        return stype.numerical

    # rule 0: to categorical stype
    if any([kw in col_name.lower() for kw in categorical_keywords]):
        if verbose and guess_type != stype.categorical:
            print(
                f"[rule 0]: {prefix}Inferred {col_name} from {guess_type} as categorical")

        return stype.categorical

    return guess_type


def custom_rule_1_infer(
    df: pd.DataFrame,
    col_name: str,
    guess_type: stype,
    prefix: str = "",
    verbose: bool = False,
) -> stype:
    """ for text_embedded or numerical columns
    if the column is too sparse, we will convert it to categorical
    """
    if guess_type == stype.text_embedded or guess_type == stype.numerical:
        unique_value = len(df[col_name].unique())
        count_value = (~df[col_name].isna()).sum()
        if unique_value * 1.0 / count_value < 0.01:
            # minimum average frequence is 100.
            if verbose:
                print(
                    f"[rule 1]: {prefix}Inferred {col_name} from {guess_type} as categorical")
            return stype.categorical
        return guess_type
    else:
        return guess_type


