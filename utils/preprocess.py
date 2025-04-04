import math
import os
import re
import wordninja
import numpy as np
import pandas as pd

from relbench.base import Database
from torch_frame import stype
from torch_frame.utils import infer_df_stype
from typing import Dict, Optional, Tuple, List

from utils.resource import get_keyword_model
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

        col_type_in_table = basic_infer_stype(
            df,
            inferred_col_to_stype,
            table_name,
            verbose)
        
        # add a rule, if it is primary key or foreign key, just categorical data
        if table.pkey_col:
            col_type_in_table[table.pkey_col] = stype.categorical
            
        for fk in table.fkey_col_to_pkey_table.keys():
            col_type_in_table[fk] = stype.categorical
        
        inferred_col_to_stype_dict[table_name] = col_type_in_table
        
    return inferred_col_to_stype_dict


# rule 0
# based on the column name
# we predefined some
numerical_keywords = [
    'count', 'num', 'amount', 'total', 'length', 'height', 'value', 'rate',  'number',
    'score', 'level', 'size', 'price', 'percent', 'ratio', 'volume', 'index', 'avg', 'max', 'min', 'age'
]

categorical_keywords = [
    'type', 'category', 'class', 'label', 'status', 'code', 'id', 'guid',
    'region', 'zone', 'flag', 'is_', 'has_', 'mode', 'duration','url', 'pid'
]

text_keywords = [
    'description', 'comments', 'content', 'name', 'review', 'message', 'note', 'query', 'summary'
]



def tokenize_identifier(identifier):
    # Try to split snake_case and kebab-case first
    if '_' in identifier:
        return identifier.split('_')
    elif '-' in identifier:
        return identifier.split('-')
    else:
        # Fallback to CamelCase or plain
        tokens = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', identifier)
        # Use wordninja to further split tokens if needed
        
        final_tokens = []
        for token in tokens:
            final_tokens.extend(wordninja.split(token))
        return final_tokens + tokens


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
    tokens = [i.lower() for i in tokenize_identifier(col_name)]
    if set(tokens) & set(text_keywords):
        
        if verbose and guess_type != stype.text_embedded:
            print(
                f"[rule 0]: {prefix} Inferred {col_name} from {guess_type} as text_embedded")
        
        return stype.text_embedded

    # rule 0: to numerical stype
    if set(tokens) & set(numerical_keywords):
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
    if set(tokens) & set(categorical_keywords):
        if verbose and guess_type != stype.categorical:
            print(
                f"[rule 0]: {prefix} Inferred {col_name} from {guess_type} as categorical")

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
                    f"[rule 1]: {prefix} Inferred {col_name} from {guess_type} as categorical")
            return stype.categorical
        return guess_type
    else:
        return guess_type


