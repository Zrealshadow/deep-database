import networkx as nx
import numpy as np
import pandas as pd
import torch
import os
import time

from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType
from torch_geometric.utils import sort_edge_index


from relbench.base import Database, Dataset

from utils.util import remove_pkey_fkey, to_unix_time
from torch_frame.data import Dataset as TFDataset
from torch_frame.config import TextEmbedderConfig
from torch_frame import stype

from typing import Dict, Optional, List, Tuple, Optional, Union, Any

from dataclasses import dataclass
from collections import defaultdict, deque


@dataclass
class DBIndex:
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
class HomoGraph:
    """
    (row_i, col_i) represents the i-th edge in the graph.
    """
    row: List[int]
    col: List[int]
    dbindex: DBIndex


def make_homograph_from_db(
    db: Database,
    verbose: bool = True,
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
            fkey_index = pd.Series(
                np.arange(len(pkey_index)), index=pkey_index.index)

            # Filter missing value
            pkey_index = pkey_index[mask].astype(int)
            fkey_index = fkey_index[mask]

            # convert to node id
            pkey_gid = dbindex.get_global_ids(pkey_table_name, pkey_index)
            fkey_gid = dbindex.get_global_ids(table_name, fkey_index)

            # fkey -> pkey edges
            row.extend(fkey_gid)
            col.extend(pkey_gid)

            # pkey -> fkey edges
            row.extend(pkey_gid)
            col.extend(fkey_gid)

            if verbose:
                print(
                    f"table {table_name} -> table {pkey_table_name} has {len(pkey_gid)} edges")

    return HomoGraph(row, col, dbindex)


def identify_entity_table(
    db: Database,
) -> List[str]:
    """following the mannul rule to identify the entity table
    return: List of table name

    based on assumption: *relation table has more rows than entity table*
    algorithm:
        1. calculate the average row of the tables
        2. return the table with row > average
    """
    table_rows = {table_name: table.df.shape[0]
                  for table_name, table in db.table_dict.items()}
    ave_row = sum(table_rows.values()) / len(table_rows)

    # check the foreign key number
    table_names = [table_name for table_name,
                   row in table_rows.items() if row < ave_row]
    tables = []
    for table_name in table_names:
        fkey_dict = db.table_dict[table_name].fkey_col_to_pkey_table
        if len(fkey_dict) <= 1:
            tables.append(table_name)
    return tables


@dataclass
class TableHopMatrix:
    graph: Dict[str, List[str]]
    # table_name -> [table_name]

    def search_tables(
        self,
        start_table: str,
        hop_threshold: int
    ) -> List[str]:
        """Search tables above hop_threshold from the start_table.
        Return: List of table names
        """
        assert start_table in self.graph, f"start_table {start_table} not in the graph"
        queue = deque([(start_table, 0)])
        visited = set(start_table)
        result = []

        while queue:
            node, hops = queue.popleft()
            if hops >= hop_threshold:
                result.append(node)

            for neighbor in self.graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, hops + 1))

        return result


def generate_hop_matrix(
    db: Database,
):
    graph = defaultdict(list)
    for table_name, table in db.table_dict.items():
        for _, pkey_table in table.fkey_col_to_pkey_table.items():
            graph[table_name].append(pkey_table)
            graph[pkey_table].append(table_name)

    return TableHopMatrix(graph)


def build_pyg_hetero_graph(
    db: Database,
    col_type_dict: Dict[str, Dict[str, stype]],
    text_embedder_cfg: Optional[TextEmbedderConfig] = None,
    cache_dir: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[HeteroData, Dict[str, Dict[str, Dict[StatType, Any]]]]:
    """ Build heterogeneous graph from the database
    """

    start_cpu_time = time.time()
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    data = HeteroData()
    col_stats_dict = {}
    for table_name, table in db.table_dict.items():
        df = table.df
        # (important for foreignKey value) Ensure the pkey is consecutive
        if table.pkey_col is not None:
            assert (df[table.pkey_col].values == np.arange(len(df))).all()

        col_to_stype = col_type_dict[table_name]

        # remove pkey, fkey
        remove_pkey_fkey(col_to_stype, table)

        if len(col_to_stype) == 0:

            if len(table.fkey_col_to_pkey_table.keys()) == 2:
                # just a relationship table, we can add edge
                col_name_x, col_name_y = table.fkey_col_to_pkey_table.keys()
                table_name_x, table_name_y = table.fkey_col_to_pkey_table.values()

                if verbose:
                    print(
                        f"-----> Build edge between {table_name_x} and {table_name_y}")

                mask = df[col_name_x].notnull() & df[col_name_y].notnull()
                index_x = torch.from_numpy(
                    df[col_name_x][mask].astype(int).values)
                index_y = torch.from_numpy(
                    df[col_name_y][mask].astype(int).values)

                # fkey -> pkey edges
                edge_index = torch.stack([index_x, index_y], dim=0)
                edge_type = (table_name_x, f"edge_{table_name}", table_name_y)
                data[edge_type].edge_index = sort_edge_index(edge_index)

                # pkey -> fkey edges.
                # "rev_" is added so that PyG loader recognizes the reverse edges
                edge_index = torch.stack([index_y, index_x], dim=0)
                edge_type = (
                    table_name_y, f"rev_edge_{table_name}", table_name_x)
                data[edge_type].edge_index = sort_edge_index(edge_index)
                continue

            else:
                # for example, relationship table which only contains pkey and fkey
                raise KeyError(f"{table_name} has no column to build graph")

        path = (
            None if cache_dir is None else os.path.join(
                cache_dir, f"{table_name}.pt")
        )

        if verbose:
            print(f"-----> Materialize {table_name} Tensor Frame")
        dataset = TFDataset(
            df=df,
            col_to_stype=col_to_stype,
            col_to_text_embedder_cfg=text_embedder_cfg,
        ).materialize(path=path)

        data[table_name].tf = dataset.tensor_frame
        col_stats_dict[table_name] = dataset.col_stats

        # Add time attribute
        if table.time_col is not None:
            data[table_name].time = torch.from_numpy(
                to_unix_time(df[table.time_col])
            )

        # Add edges normal edges
        for fkey_col_name, pkey_table_name in table.fkey_col_to_pkey_table.items():
            pkey_index = df[fkey_col_name]
            # Filter out dangling foreign keys
            mask = ~pkey_index.isna()
            fkey_index = torch.arange(len(pkey_index))

            # filter dangling foreign keys:
            pkey_index = torch.from_numpy(pkey_index[mask].astype(int).values)
            fkey_index = fkey_index[torch.from_numpy(mask.values)]

            # fkey -> pkey edges
            edge_index = torch.stack([fkey_index, pkey_index], dim=0)
            edge_type = (table_name, f"f2p_{fkey_col_name}", pkey_table_name)
            data[edge_type].edge_index = sort_edge_index(edge_index)

            # pkey -> fkey edges.
            # "rev_" is added so that PyG loader recognizes the reverse edges
            edge_index = torch.stack([pkey_index, fkey_index], dim=0)
            edge_type = (pkey_table_name,
                         f"rev_f2p_{fkey_col_name}", table_name)
            data[edge_type].edge_index = sort_edge_index(edge_index)

    data.validate()
    end_cpu_time = time.time()
    cpu_time_cost = end_cpu_time - start_cpu_time
    if verbose:
        print(f"Build pyg hetero graph takes {cpu_time_cost:.6f} seconds")
    return data, col_stats_dict
