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

# ----------------- Tokenize attribute -----------------


@dataclass
class TokenizedDatabase:
    table_dict: Dict[str, List[List[str]]]
    prefix_dict: Dict[str, str]
    # table_name-col_name -> prefix

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
    prefix_dict = {}

    # prefix -> table_name -> col_name
    for table_name in db.table_dict.keys():
        cols = db.table_dict[table_name].df.columns
        for col in cols:
            prefix_dict[f"{table_name}-{col}"] = f"{len(prefix_dict)}"

    # table_name -> List[row_attributes]
    for table_name, table in db.table_dict.items():
        df = table.df.copy()
        pkey_col = db.table_dict[table_name].pkey_col

        if verbose:
            print(f"----------------> Tokenizing {table_name} each column")

        file_path = os.path.join(
            cache_path_dir, f"{table_name}.npy") if cache_path_dir else None

        if file_path and os.path.exists(file_path):
            if verbose:
                print(f"-> Load tokenized data from {file_path}")
            table_dict[table_name] = np.load(
                file_path, allow_pickle=True).tolist()
            continue

        # if not exists, tokenize the data
        for col_name in df.columns:
            if col_name not in col_to_stype[table_name] or col_name == pkey_col:
                # skip pkey, which is duplicated with fkey
                if verbose:
                    print(f"Skip pkey column {table_name}-{col_name}")
                continue

            # need to convert to object type, otherwise apply list will raise error
            df[col_name] = df[col_name].astype(object)
            guess_type = col_to_stype[table_name][col_name]
            prefix = prefix_dict[f"{table_name}-{col_name}"]

            if guess_type != stype.text_embedded \
                    and guess_type != stype.numerical:
                # tokenized the other data
                # -> table_name-col_name-value
                not_nan_mask = df[col_name].notna()
                df.loc[not_nan_mask, col_name] = df.loc[not_nan_mask, col_name].apply(
                    lambda x: [f"{prefix}-{x}"])

                # convert to list, for conveniece of subsequent aggregation

            if guess_type == stype.text_embedded:
                print(f"Tokenize text column {table_name}-{col_name}")
                __tokenize_text_column(
                    df, col_name, table_name, prefix, verbose)

            if guess_type == stype.numerical:
                # print(f"Tokenize column column {table_name}-{col_name}")
                __tokenize_numerical_column(
                    df, col_name, table_name, prefix, verbose)

        # cache the tokenized data based on table_name

        # convert the row of the table to list of attributes
        if verbose:
            print(
                f"-> Manage the tuple in table {table_name} to list of attributes ")

        pkys_attributes_doc = []
        # remove the pky column
        other_columns = df.columns.drop(pkey_col).tolist()
        for row in df[other_columns].values:
            row = row[~pd.isna(row)].tolist()
            try:
                attributes = [x for xx in row for x in xx]
            except:
                print(row)
                raise ValueError
            pkys_attributes_doc.append(attributes)

        if file_path:
            np.save(file_path, np.array(pkys_attributes_doc, dtype='object'))

        table_dict[table_name] = pkys_attributes_doc

    return TokenizedDatabase(table_dict=table_dict, prefix_dict=prefix_dict)


def __tokenize_text_column(
    df: pd.DataFrame,
    col_name: str,
    table_name: str,
    prefix: str,
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
    prefix = f"{prefix}-"
    sentence_to_kws = {st: [prefix + kw for kw, _ in kws]
                       for st, kws in sentence_to_kws.items()}

    # apply the token to the column
    not_nan_mask = df[col_name].notna()

    df.loc[not_nan_mask, col_name] = df.loc[not_nan_mask, col_name].map(
        sentence_to_kws)

    if verbose:
        print(
            f"=> {table_name} col @{col_name} has {len(series)} records, has {len(sentence_set)} unique sentences")
    return


def __tokenize_numerical_column(
    df: pd.DataFrame,
    col_name: str,
    table_name: str,
    prefix: str,
    verbose: bool = False,
) -> None:
    """ for numerical data, we try to cluster and bin it convert it to categroical data.
    """

    n = (~df[col_name].isna()).sum()
    not_nan_mask = df[col_name].notna()
    series = df[col_name]
    binned = pd.Series(index=series.index, dtype='object')
    binned[~not_nan_mask] = np.NaN

    # Step 1. determine the bin number
    if n > 1_000:
        # rice Rule
        bin_num = math.ceil(2 * n ** (1/3))
    else:
        # n < 1000,
        # Sturge's Rule
        bin_num = math.ceil(1 + math.log2(n))

    # Step 2. bin the outlier
    q1 = df[not_nan_mask][col_name].quantile(0.1)
    q2 = df[not_nan_mask][col_name].quantile(0.9)
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
    binned = binned.apply(lambda x: [f"{prefix}-{x}"])
    df[col_name] = binned
    # wrapper with list, for the convenience of subsequent aggregation
    return
