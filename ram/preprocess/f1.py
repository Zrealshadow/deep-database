# %%
# %cd ..
import torch
import math
import torch_frame
import copy
from tqdm import tqdm
from utils.data import DatabaseFactory
from utils.builder import build_pyg_hetero_graph
from utils.resource import get_text_embedder_cfg
from utils.util import load_col_types
from utils.document import generate_document_given_table
from utils.builder import identify_entity_table
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# %%
db = DatabaseFactory().get_db("f1")

# %%
from utils.builder import make_homograph_from_db
homoGraph = make_homograph_from_db(db, verbose=True)

# %%
from utils.preprocess import infer_type_in_db
from utils.tokenize import tokenize_database


# %%
col_type_dict = infer_type_in_db(db, verbose=True)

# %%
tk_db = tokenize_database(db, col_type_dict, './ram/tmp_docs/rel-f1', True)

# %%
from utils.document import generate_document_given_table
from utils.builder import identify_entity_table
from utils.builder import generate_hop_matrix

# %%
entity_tables = identify_entity_table(db)
entity_tables

# %%
# generated the documents and build the retrieval index
entity_to_docs = {}
walk_length = 10
round = 8
for entity in entity_tables:
   _, entity_to_docs[entity] = generate_document_given_table(
        homoGraph, 
        tk_db, 
        entity, 
        walk_length=walk_length, 
        round = round, 
        verbose=True
    )

# %%
import bm25s
tmp_data_dir = "f1"
entity_to_retriver = {}


for entity, docs in entity_to_docs.items():
    retriever = bm25s.BM25(backend="numba")
    retriever.index(docs)
    retriever.activate_numba_scorer()
    entity_to_retriver[entity] = retriever

# save the retriever
for entity, retriever in entity_to_retriver.items():
    retriever.save(f"./ram/tmp/{tmp_data_dir}/{entity}_retriever_bm25")

# # load the retriever
# entity_to_retriver = {}
# for entity in entity_tables:
#     path = f"./ram/tmp/{tmp_data_dir}/{entity}_retriever_bm25"
#     retriever = bm25s.BM25.load(path)
#     retriever.activate_numba_scorer()
#     entity_to_retriver[entity] = retriever
#     print(f"load {path}")

# %%
# generated the documents and build the retrieval index
entity_to_docs = {}
walk_length = 10
round = 8
entity_to_docs = {}
entity_candidate_pkys = {}
# for each
for entity in entity_tables:
    n = len(db.table_dict[entity].df)
    # sample_size = n // 10
    sample_size = n
    entity_candidate_pkys[entity], entity_to_docs[entity] = generate_document_given_table(
        homoGraph, 
        tk_db, 
        entity, 
        walk_length=walk_length, 
        round = round,
        sample_size = sample_size,
        verbose=True
    )

# %%
# Add the cross-table edges,
# first we want to find the multi-hop entity pairs
hop_matrix = generate_hop_matrix(db)
edge_candidates_pairs = []
for entity in entity_tables:
    for entity2 in entity_tables:
        if entity == entity2:
            continue
        
        if entity2 not in hop_matrix.graph[entity]:
            # not one hop
            edge_candidates_pairs.append((entity, entity2))
edge_candidates_pairs

# %%
import numpy as np
topn = 10
edge_dict = {}
# (src_table, des_table) -> edge 2-D array
for entity, retrieve_entity in edge_candidates_pairs:

    # retrieve the related docs
    entity_query_docs = entity_to_docs[entity]
    entity_query_pkys = entity_candidate_pkys[entity]
    retriever = entity_to_retriver[retrieve_entity]
    
    related_pkys, scores = retriever.retrieve(entity_query_docs, k = topn, n_threads = 24)
    
    score_np = np.array(scores)
    related_pkys_np = np.array(related_pkys)
    threshold = score_np.mean() + 2*scores.std()
    # threshold = score_np.mean() + scores.std()
    # threshold = score_np.mean()
    # threshold = score_np.mean() + 3* scores.std()
    # Get indices where the score is above the threshold
    mask = score_np > threshold

    # Apply the mask
    filtered_cols = related_pkys_np[mask]

    # Generate the corresponding query entities
    entity_query_pkys = np.array(entity_query_pkys)  # shape [n]

    # Repeat each query item the number of True values per row in the mask
    row_repeats = mask.sum(axis=1)  # how many times to repeat each query
    filtered_rows = np.repeat(entity_query_pkys, row_repeats)
    
    
    filtered_edge = np.stack([filtered_rows, filtered_cols], axis=1)
    # added edge
    num_edges = filtered_rows.shape[0]
    edge_dict[(entity, retrieve_entity)] = filtered_edge
    print(f"Add cross table edges #{num_edges} between {entity} and {retrieve_entity}")

# %%
# (src_table, des_table) -> edge 2-D array
npz_data = {
    f"{src}-{dst}": edge_array
    for (src, dst), edge_array in edge_dict.items()
}

path = f"./ram/edges/f1-edges.npz"
np.savez(path, **npz_data)

# %% [markdown]
# Self entity correlation, which generate positive pairs in the constrastive learning.

# %%
import numpy as np
topn = 11
# the most related doc should be itself, so we need to retrieve topn + 1
positive_pool_dict = {}
# entity -> positive candidate, padding the non-value
threshold = 0.6
batch_size = 2048
recall_score_dict = {}
related_pkys_dict = {}
for entity, retriever in entity_to_retriver.items():
    # retrieve the related docs
    entity_query_docs = entity_to_docs[entity]
    entity_query_pkys = entity_candidate_pkys[entity]
    score_np = []
    related_pkys_np = []
    print(f"--------> {entity}")
    for batch_idx in tqdm(range(0, len(entity_query_docs), batch_size)):
        batch_query_docs = entity_query_docs[batch_idx:batch_idx + batch_size]
        related_pkys, scores = retriever.retrieve(batch_query_docs, k = topn, n_threads=-1)
        score_np.append(np.array(scores))
        related_pkys_np.append(np.array(related_pkys))
    
    score_np = np.concatenate(score_np, axis = 0)
    related_pkys_np = np.concatenate(related_pkys_np, axis = 0)
    recall_score_dict[entity] = score_np
    related_pkys_dict[entity] = related_pkys_np
    print(f"Finish Retrieval in {entity}")

# %%
threshold = 0.75
positive_pool_dict={}
cnt = 0
import copy
for entity, score_np in recall_score_dict.items():
    entity_query_docs = entity_to_docs[entity]
    mask = score_np > (score_np[:,[0]] * threshold)
    related_pkys_np = copy.deepcopy(related_pkys_dict[entity])
    related_pkys_np[~mask] = -1
    rows_num = np.sum(mask, axis = 1)
    # except itself, still has similar docs
    rows_mask = rows_num > 1
    positive_pool =  related_pkys_np[rows_mask]
    positive_pool_dict[entity] = positive_pool
    num = np.sum(related_pkys_np != -1) - related_pkys_np.shape[0]
    cnt += num
    print(f"Generate positive pools #{len(positive_pool)}, num {num} original candidate {len(entity_query_docs)} in {entity} table")

print(f'total positive pairs {cnt}')


# %%
path = "./samples/f1-samples.npz"
np.savez(path, **positive_pool_dict)

# %%



