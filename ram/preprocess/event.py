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
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# cache_dir = "./data/rel-trial-tensor-frame"

# %%
db = DatabaseFactory.get_db("event", with_text_compress=False)

# %%
from utils.document import generate_document_given_table
from utils.builder import identify_entity_table
entity_tables = identify_entity_table(db)
entity_tables

# %%
edge_candidates_pairs = []
edge_candidates_pairs.extend([("events", "users"), ("users", "events")])

# %%
# homoGraph
from utils.builder import HomoGraph, make_homograph_from_db
homoGraph = make_homograph_from_db(db, verbose=True)

# %%
from utils.preprocess import infer_type_in_db
from utils.tokenize import tokenize_database
col_type_dict = infer_type_in_db(db, True)

# %%
# check all col types
for table_name, col_types in col_type_dict.items():
    print(f"Table {table_name}")
    for col, type_ in col_types.items():
        print(f"{col}: {type_}")
    print("*"*40)

# %%
tk_db = tokenize_database(db, col_type_dict, './ram/tmp_docs/rel-events', True)

# %%
# generated the documents and build the retrieval index
# entity_to_docs = {}
# walk_length = 10
# round = 10
# for entity in entity_tables:
#    _, entity_to_docs[entity] = generate_document_given_table(
#         homoGraph, 
#         tk_db, 
#         entity, 
#         walk_length=walk_length, 
#         round = round, 
#         verbose=True
#     )

# %%
# # temporarily save the index
# import bm25s
# entity_to_retriver = {}
# for entity, docs in entity_to_docs.items():
#     retriever = bm25s.BM25(backend="numba")
#     retriever.index(docs)
#     retriever.activate_numba_scorer()
#     entity_to_retriver[entity] = retriever

# # save the retriever
# for entity, retriever in entity_to_retriver.items():
#     retriever.save(f"./ram/tmp/event/{entity}_retriever_bm25")

# %%
import bm25s
entity_to_retriver = {}

# load the retriever
entity_to_retriver = {}
for entity in entity_tables:
    path = f"./ram/tmp/event/{entity}_retriever_bm25"
    retriever = bm25s.BM25.load(path)
    retriever.activate_numba_scorer()
    entity_to_retriver[entity] = retriever
    print(f"load {path}")

# %%
# resample the candidate docs, and retrieve the related docs in the bm25 retrievers
# generated the documents and build the retrieval index
walk_length = 8
round = 10
entity_to_docs = {}
entity_candidate_pkys = {}
# for each
for entity in entity_tables:
    n = len(db.table_dict[entity].df)
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
import numpy as np
topn = 20
edge_dict = {}
batch_size = 2048
# (src_table, des_table) -> edge 2-D array
for entity, retrieve_entity in edge_candidates_pairs:

    # retrieve the related docs
    entity_query_docs = entity_to_docs[entity]
    entity_query_pkys = entity_candidate_pkys[entity]
    retriever = entity_to_retriver[retrieve_entity]
    
    print(f"--------> {entity} ---- {retrieve_entity}")
    score_np = []
    related_pkys_np = []
    for batch_idx in tqdm(range(0, len(entity_query_docs), batch_size)):
        batch_query_docs = entity_query_docs[batch_idx:batch_idx+batch_size]
        related_pkys, scores = retriever.retrieve(batch_query_docs, k = topn, n_threads = -1)
        score_np.append(np.array(scores))
        related_pkys_np.append(np.array(related_pkys))
    
    score_np = np.concatenate(score_np, axis = 0)
    related_pkys_np = np.concatenate(related_pkys_np, axis = 0)
    
    # threshold = score_np.mean() + 2*scores.std()
    # threshold = score_np.mean() 
    # threshold = score_np.mean() + 1*score_np.std()
    threshold = score_np.mean() + 3*score_np.std()
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

# path = f"./ram/edges/rel-event-edges.npz"
# path = f"./ram/edges/rel-event-edges-thresh-mean.npz"
# path = f"./ram/edges/rel-event-edges-thresh-mean-std.npz"
path = f"./ram/edges/rel-event-edges-thresh-mean-3std.npz"
np.savez(path, **npz_data)

# %%
# resample the candidate docs, and retrieve the related docs in the bm25 retrievers
# generated the documents and build the retrieval index
walk_length = 10
round = 10
entity_to_docs = {}
entity_candidate_pkys = {}
# for each
for entity in entity_tables:
    n = len(db.table_dict[entity].df)
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
# self-entity correlation
# which can generate the positive pairs in the contrastive learning
entity_topn = {
    "events": 21,
    "users": 21,
}
# the most related doc should be itself, so we need to retrieve topn + 1
positive_pool_dict = {}
# entity -> positive candidate, padding the non-value
# threshold = 0.7
threshold = 0.9
batch_size = 2048
for entity, retriever in entity_to_retriver.items():
    # retrieve the related docs
    topn = entity_topn[entity]
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
    # Get indices where the score is above the threshold
    # the first one is the most related one, should be itself
    mask = score_np > (score_np[:,[0]] * threshold)
    # add padding for those non-related docs which is filtered out.
    related_pkys_np[~mask] = -1
    rows_num = np.sum(mask, axis = 1)
    # except itself, still has similar docs
    rows_mask = rows_num > 1
    positive_pool = related_pkys_np[rows_mask]
    
    positive_pool_dict[entity] = positive_pool
    print(f"Generate positive pools #{len(positive_pool)}, original candidate {len(entity_query_docs)} in {entity} table")

# %%
# path = "./samples/rel-event-samples-thresh05.npz"
# path = "./samples/rel-event-samples-thresh06.npz"
# path = "./samples/rel-event-samples-thresh08.npz"
path = "./samples/rel-event-samples-thresh09.npz"
np.savez(path, **positive_pool_dict)

# %%



