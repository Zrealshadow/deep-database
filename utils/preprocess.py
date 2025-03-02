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
    'region', 'zone', 'flag', 'is_', 'has_', 'mode',
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
        if verbose:
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
    else:
        return guess_type


# ----------------- Tokenize attribute -----------------

@dataclass
class TokenizedDatabase(object):
    table_dict = Dict[str, List[List[str]]]

    def get_tuple_attributes_set(self, table_name: str, pky_idx: int) -> List[str]:
        # "word" -> "table_name-col_name-value"
        table = self.table_dict[table_name]
        assert pky_idx < len(table)
        return table[pky_idx]


def tokenize_database(
    db: Database,
    col_to_stype: Optional[Dict[str, Dict[str, stype]]] = None,
    cache_path_dir: Optional[str] = None,
    verbose: bool = False,
) -> TokenizedDatabase:
    """
    return 
        -  tokenized dict {table_name -> List[row_attributes]}
    """
    table_dict = {}
    # table_name -> List[row_attributes]
    for table_name, df_ in db.table_dict.items():
        df = df_.copy()

        if verbose:
            print(f"----------------> Tokenizing {table_name} each column")

        file_path = os.path.join(
            cache_path_dir, f"{table_name}.npy") if cache_path_dir else None

        if file_path and os.path.exists(file_path):
            if verbose:
                print(f"-> Load tokenized data from {file_path}")
            table_dict[table_name] = np.load(file_path, allow_pickle=True)
            continue

        # if not exists, tokenize the data
        for col_name in tqdm(df.columns, leave=False):
            if col_name not in col_to_stype[table_name]:
                continue

            guess_type = col_to_stype[table_name][col_name]

            if guess_type != stype.text_embedded \
                    and guess_type != stype.numerical:
                # tokenized the other data
                # -> table_name-col_name-value
                no_nan_mask = df[col_name].notna()
                df[col_name][no_nan_mask] = df[col_name][no_nan_mask].apply(
                    lambda x: ["{table_name}-{col_name}-{x}"])

                # convert to list, for conveniece of subsequent aggregation

            if guess_type == stype.text_embedded:
                tokenize_text_column(
                    df, col_name, table_name, verbose)

            if guess_type == stype.numerical:
                tokenize_numerical_column(df, col_name, table_name, verbose)

        # cache the tokenized data based on table_name

        # convert the row of the table to list of attributes
        if verbose:
            print("-> Manage the tuple to list of attributes")

        pkys_attributes_doc = []
        for row in df.values:
            row = row[~pd.isna(row)].tolist()
            attributes = [x for xx in row for x in xx]
            pkys_attributes_doc.append(attributes)

        if file_path:
            np.save(file_path, np.array(pkys_attributes_doc, dtype='object'))

        table_dict[table_name] = pkys_attributes_doc

    return TokenizedDatabase(table_dict=table_dict)


def tokenize_text_column(
    df: pd.DataFrame,
    col_name: str,
    table_name: str = "",
    verbose: bool = False,
) -> None:
    """for text type data, basically, text type data is more diverse than numerical data and categorical data
    there are several steps to process:

    --- 1. using pre-trained model to extract the keywords from text.
    --- 2. these keywords are concatenated with the column name to form the token in "doc"

    in KeyBERT model, we set the setting to default, only extract one-gram keywors for simplicity.

    why just extract keywords from this keywords ?
    -- reduces noise by focusing only on meaningful and frequent concepts.
    -- creates a more compact document with relevant terms.
    """
    sentence_to_kws = {}

    kw_model = get_keyword_model()
    series = df[col_name]
    sentences = series[series.notna()].tolist()

    # don't repeated extract the same sentence
    sentence_set = list(set(sentences))

    # convert the sentences to keyword set
    batch_size = 2048
    sentence_to_kws = {}
    for i in range(0, len(sentence_set), batch_size):
        batch = sentence_set[i:i+batch_size]
        kws = kw_model.extract_keywords(batch)
        sentence_to_kws.update(dict(zip(batch, kws)))

    # each keyword is like this [('keyword1', 0.9), ('keyword2', 0.8), ('keyword3', 0.7)]

    # there we just take all keyword equally, remove the score
    prefix = f"{table_name}-{col_name}-"
    sentence_to_kws = {st: [prefix + kw for kw, _ in kws]
                       for st, kws in sentence_to_kws.items()}

    # apply the token to the column
    not_nan_mask = df[col_name].notna()

    df[col_name][not_nan_mask] = df[col_name][not_nan_mask].map(
        sentence_to_kws)

    if verbose:
        print(
            f"=> {table_name} col @{col_name} has {len(series)} records, has {len(sentence_set)} unique sentences")
    return


def tokenize_numerical_column(
    df: pd.DataFrame,
    col_name: str,
    table_name: str = "",
    verbose: bool = False,
) -> None:
    """ for numerical data, we try to cluster and bin it convert it to categroical data.
    """

    n = (~df[col_name].isna()).sum()
    not_nan_mask = df[col_name].notna()
    series = df[col_name][not_nan_mask]

    binned = pd.Series(index=series.index, dtype='object')

    # Step 1. determine the bin number
    if n > 1_000:
        # rice Rule
        bin_num = math.ceil(2 * n ** (1/3))
    else:
        # n < 1000,
        # Sturge's Rule
        bin_num = math.ceil(1 + math.log2(n))

    # Step 2. bin the outlier
    q1 = series.quantile(0.1)
    q2 = series.quantile(0.9)
    upper_bound = q2 + 1.5 * (q2 - q1)
    lower_bound = q1 - 1.5 * (q2 - q1)

    upper_outlier_mask = series > upper_bound
    lower_outlier_mask = series < lower_bound
    normal_mask = ~upper_outlier_mask & ~lower_outlier_mask
    normal_mask = normal_mask & series.notna()

    upper_outlier_label = f"larger than {upper_bound}"
    lower_outlier_label = f"smaller than {lower_bound}"

    binned[upper_outlier_mask] = upper_outlier_label
    binned[lower_outlier_mask] = lower_outlier_label

    # Step 3. bin the normal data
    _, bin_edges = pd.cut(series[normal_mask], bins=bin_num,
                          labels=False, retbins=True, include_lowest=True)
    bin_labels = [
        f"{bin_edges[i]:.0f}-{bin_edges[i+1]:.0f}" if bin_edges[i].is_integer() and bin_edges[i+1].is_integer()
        else f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}"
        for i in range(len(bin_edges) - 1)
    ]

    binned[normal_mask] = pd.cut(
        series[normal_mask], bins=bin_edges, labels=bin_labels)

    if verbose:
        print(
            f"Bin {table_name}.{col_name} to {bin_num} bins, convert numerical data to categorical data")

    df[col_name][not_nan_mask] = binned
    df[col_name][not_nan_mask] = df[col_name][not_nan_mask].apply(
        lambda x: [f"{table_name}-{col_name}-{x}"])
    # wrapper with list, for the convenience of subsequent aggregation
