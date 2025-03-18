import torch
from torch_cluster import random_walk

from utils.builder import HomoGraph
from utils.tokenize import TokenizedDatabase

from functools import reduce

from typing import List
from tqdm import tqdm

def generate_document_given_table(
    g: HomoGraph,
    tk_db: TokenizedDatabase,
    table_name: str,
    walk_length: int = 20,
    round: int = 10,
    p: float = 10,
    q: float = 0.5,
    sample_size: int = -1,
    verbose: bool = False,
):
    """
    return:
    - List[int]: List of primary keys of the target nodes.
    - List[List[str]]: List of documents. Each document is a list of strings.
    """
    
    assert table_name in g.dbindex.table_gid_offset, "Table not found in the database."
    assert table_name in tk_db.table_dict, "Table not found in the tokenized database."

    row = torch.LongTensor(g.row)
    col = torch.LongTensor(g.col)

    # default walk_length is the number of table
    walk_length = min(walk_length, len(tk_db.table_dict))

    # collect target node pkys, sample_size
    n = len(tk_db.table_dict[table_name])
    if sample_size == -1:
        pkys = torch.arange(n).tolist()
    else:
        pkys = torch.randperm(n)[:sample_size].tolist()
    
    gids = g.dbindex.get_global_ids(table_name, pkys)
    gids = torch.LongTensor(gids)
    
    walks = []
    for _ in range(round):
        walk = random_walk(row, col, gids, walk_length, p, q)

        # walk -> [num_target_node, walk_length]
        walk = walk.unsqueeze(1)
        # walk -> [num_target_node, 1, walk_length]
        walks.append(walk)
    
    walks = torch.concatenate(walks, dim=1)
    # [num_target_node, round, walk_length]

    if verbose:
        print(f"- Walks for table {table_name} - shape {walks.shape}")

    
    docs = []
    # convert the walks to the tokens
    for idx in tqdm(range(walks.shape[0]), leave=False):
        node_bags = walks[idx].tolist() # [round, walk_length]
        # for each walks, remove repeated node to construct subgraph.
        # make sure in each round of sample, the node is unqiue.
        # doc_node_ids = reduce(lambda a, b: list(set(a)) + list(set(b)), node_bags)
        doc_node_ids = [i for walk in node_bags for i in list(set(walk))]
        # [node_ids]
        table_pks = g.dbindex.get_tuple_positions(doc_node_ids)
        # [(table_name, pky)]
        
        tuple_list = [tk_db.get_tuple_attributes_set(table_name, pky) for table_name, pky in table_pks]
        doc = reduce(lambda a, b: a + b, tuple_list)
        docs.append(doc)
    
    return pkys, docs

