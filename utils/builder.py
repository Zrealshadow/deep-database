import networkx as nx
import numpy as np
import pandas as pd
import torch

from relbench.base import Database
from torch_frame.config import TextEmbedderConfig
from torch_frame import stype

from typing import Dict, Optional, List, Tuple, Optional, Union

from dataclasses import dataclass
from collections import OrderedDict


@dataclass
class DBIndex(object):
    """
    Data structure to mapping between node global id and tuples in the database.
    """

    table_gid_offset: Dict[str, int]
    # table_name -> scope
    gid_table: List[str]

    def get_tuple_positions(self, global_ids: Union[int, List[int]]) -> Union[Tuple[str, int], List[Tuple[str, int]]]:
        if isinstance(global_ids, List):
            max_gid = max(global_ids)
            assert max_gid < len(self.gid_table)
            def get_offset(
                gid): return self.table_gid_offset[self.gid_table[gid]]
            return [(self.gid_table[gid], gid - get_offset(gid)) for gid in global_ids]
        else:
            assert global_ids < len(self.gid_table)
            return (self.gid_table[global_ids], global_ids - self.table_gid_offset[self.gid_table[global_ids]])

    def get_global_ids(self, table_name: str, pkys: Union[int, List[int]]) -> Union[int, List[int]]:
        if isinstance(pkys, List):
            offset = self.table_gid_offset[table_name]
            return [offset + pky for pky in pkys]
        else:
            return self.table_gid_offset[table_name] + pkys


@dataclass
class HomoGraph(object):
    """
    (row_i, col_i) represents the i-th edge in the graph.
    """
    row: List[int]
    col: List[int]
    dbindex: DBIndex


def make_homograph_from_db(
    db: Database,
) -> HomoGraph:
    """
    Generate a homogeneous graph from the database for random-walk sampling.
    """
    # --------------- initialize node ----------------
    dbindex = DBIndex(table_gid_offset={}, gid_table=[])
    for table_name, table in db.table_dict.items():
        df = table.df

        if table.pkey_col is not None:
            assert (df[table.pkey_col].values == np.arange(len(df))).all()

        table_offset = len(dbindex.gid_table)
        # update table offset
        dbindex.table_gid_offset[table_name] = table_offset
        # update gid to table
        dbindex.gid_table.extend([table_name] * len(df))

    # --------------- initialize edge ----------------
    row, col = [], []
    for table_name, table in db.table_dict.items():
        df = table.df
        for fkey_name, pkey_table_name in table.fkey_col_to_pkey_table.items():
            pkey_index = df[fkey_name]
            # Filter out dangling foreign keys (missing value)
            mask = ~pkey_index.isna()
            fkey_index = pd.Series(np.arange(len(pkey_index)))

            # Filter missing value
            pkey_index = pkey_index[mask].astype(int)
            fkey_index = fkey_index[mask]

            # convert to node id
            pkey_gid = dbindex.get_global_ids(pkey_table_name, pkey_index)
            fkey_gid = dbindex.get_global_ids(table_name, fkey_index)

            row.extend(fkey_gid)
            col.extend(pkey_gid)

    return HomoGraph(row, col, dbindex)


