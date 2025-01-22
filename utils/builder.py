import networkx as nx
import numpy as np
import pandas as pd
import torch

from relbench.base import Database
from torch_frame.config import TextEmbedderConfig
from torch_frame import stype

from typing import Dict, Optional, Tuple


def make_pkey_fkey_edges(
    db: Database,
    col_to_stype_dict: Dict[str, Dict[str, stype]] = None,
    text_embedder_cfg: Optional[TextEmbedderConfig] = None,
    cache_dir: Optional[str] = None,
):

    gid_to_entity, entity_to_gid = {}, {}
    # node_id -> (table_name, pkey_index)
    # table_name -> { pkey_index -> node_id}

    # initialize node
    for table_name, table in db.table_dict.items():
        df = table.df
        if table.pkey_col is not None:
            assert (df[table.pkey_col].values == np.arange(len(df))).all()

        gid_init = len(gid_to_entity)
        gid_end = gid_init + len(df)
        gids = np.arange(gid_init, gid_end)
        pkey_idxs = np.arange(len(df))

        gid_to_entity.update(dict(zip(gids, pkey_idxs)))
        entity_to_gid[table_name] = dict(zip(pkey_idxs, gids))

    # initialize edge
    row, col = [], []
    # source node, target node
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
            pkey_gid = pkey_index.map(entity_to_gid[pkey_table_name]).values()
            fkey_gid = fkey_index.map(entity_to_gid[table_name]).values()

            pkey_gid = torch.LongTensor(pkey_gid)
            fkey_gid = torch.LongTensor(fkey_gid)

            # fkey -> pkey edges
            row.append(fkey_gid)
            col.append(pkey_gid)

            # pkey -> fkey edges
            row.append(pkey_gid)
            col.append(fkey_gid)

    row = torch.cat(row, dim=0)
    col = torch.cat(col, dim=0)

    return (row, col)(gid_to_entity, entity_to_gid)
