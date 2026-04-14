"""
Schema Scalability: model parameter count vs. number of tables.

Progressively expands the schema hop-by-hop from the entity table:
  hop 0 -> entity table only
  hop 1 -> + all direct FK neighbors
  hop 2 -> + their neighbors
  ...
  hop k -> full graph

ONE AIDAXFormer is constructed on the full graph. For each hop level we
count only the sub-modules that correspond to the included tables / edge
types, giving the parameter footprint the model would need for that subgraph.

Scaling modules  (sliced by included tables / edges):
  • AIDABaseFeatureEncoder   per-table feature encoders
  • AIDATableEncoder         per-table CLS embeddings  +  shared transformer
  • HeteroGraphSAGE          per-edge-type SAGEConv   +  per-node-type norms

Fixed modules (always counted at full size):
  • FusionLayer
  • head
"""

import argparse
from collections import defaultdict

from utils.util import load_col_types, setup_torch
from utils.resource import get_text_embedder_cfg
from utils.builder import build_pyg_hetero_graph
from utils.data import DatabaseFactory
from model.aida import construct_default_AIDAXFormer

setup_torch()

parser = argparse.ArgumentParser()
parser.add_argument("--tf_cache_dir", type=str, required=True)
parser.add_argument("--db_name",      type=str, required=True)
parser.add_argument("--task_name",    type=str, required=True)
parser.add_argument("--channels",           type=int, default=128)
parser.add_argument("--feat_layer_num",     type=int, default=2)
parser.add_argument("--relation_layer_num", type=int, default=2)
parser.add_argument("--feat_nhead",         type=int, default=1)
parser.add_argument("--relation_aggr",      type=str, default="sum")
args = parser.parse_args()

# ── Load full graph ───────────────────────────────────────────────────────────
db      = DatabaseFactory.get_db(db_name=args.db_name, with_text_compress=True)
dataset = DatabaseFactory.get_dataset(db_name=args.db_name)
task    = DatabaseFactory.get_task(db_name=args.db_name, task_name=args.task_name, dataset=dataset)

col_type_dict = load_col_types(cache_path=args.tf_cache_dir, file_name="col_type_dict.pkl")
data, col_stats_dict = build_pyg_hetero_graph(
    db, col_type_dict, get_text_embedder_cfg(device="cpu"),
    cache_dir=args.tf_cache_dir, verbose=False,
)

entity_table = task.entity_table

# ── Build ONE full model on the complete graph ────────────────────────────────
net = construct_default_AIDAXFormer(
    data, col_stats_dict,
    channels=args.channels,
    out_channels=1,
    feat_layer_num=args.feat_layer_num,
    feat_nhead=args.feat_nhead,
    relation_layer_num=args.relation_layer_num,
    relation_aggr=args.relation_aggr,
)


# ── Parameter-counting helpers ────────────────────────────────────────────────
def _p(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def count_subgraph_params(net, included_tables, included_edge_types):
    """
    Slice the full model and count parameters relevant to the subgraph.

    Returns (total, feat_enc, tbl_enc, relation, fusion).
    """
    included    = set(included_tables)
    edge_keys   = set(included_edge_types)  # tuples (src, rel, dst) — PyG ModuleDict uses tuple keys

    # ── AIDABaseFeatureEncoder: per-table feature encoders ────────────────────
    feat_enc = sum(
        _p(enc)
        for t, enc in net.feat_encoder.feature_encoders.items()
        if t in included
    )

    # ── AIDATableEncoder: shared transformer (fixed) + per-table CLS tokens ──
    te = net.table_encoder.shared_table_encoder
    tbl_enc  = _p(te.transformer) + _p(te.norm)       # shared overhead
    tbl_enc += sum(
        emb.numel()
        for t, emb in te.table_cls_embeddings.items()
        if t in included
    )

    # ── HeteroGraphSAGE: per-edge-type SAGEConv + per-node-type norms ─────────
    relation = 0
    if net.relation_module is not None:
        rm = net.relation_module
        for conv_layer in rm.convs:
            # conv_layer is HeteroConv; .convs is a ModuleDict keyed 'src__rel__dst'
            for key, sage_conv in conv_layer.convs.items():
                if key in edge_keys:
                    relation += _p(sage_conv)
        for norm_dict in rm.norms:
            for t, norm in norm_dict.items():
                if t in included:
                    relation += _p(norm)
    # relation += _p(net.temporal_encoder)
    # ── FusionLayer: always full (entity-centric, fixed architecture) ─────────
    fusion = _p(net.fusion_module) if net.fusion_module is not None else 0

    # ── head: always ──────────────────────────────────────────────────────────
    # total = feat_enc + tbl_enc + relation + fusion + _p(net.head)
    total = tbl_enc + relation + fusion + _p(net.head)
    return total, feat_enc, tbl_enc, relation, fusion


# ── BFS from entity table to get hop-level sets ───────────────────────────────
adjacency = defaultdict(set)
for src, _, dst in data.edge_types:
    adjacency[src].add(dst)
    adjacency[dst].add(src)

visited  = {entity_table}
hop_sets = [{entity_table}]

frontier = {entity_table}
while True:
    next_frontier = set()
    for table in frontier:
        for neighbor in adjacency[table]:
            if neighbor not in visited:
                visited.add(neighbor)
                next_frontier.add(neighbor)
    if not next_frontier:
        break
    hop_sets.append(next_frontier)
    frontier = next_frontier


def subgraph_edge_types(included_tables):
    included = set(included_tables)
    return [
        (src, rel, dst)
        for src, rel, dst in data.edge_types
        if src in included and dst in included
    ]


# ── Build BFS-ordered table list (one table at a time) ────────────────────────
bfs_ordered_tables = []
for hop_tables in hop_sets:
    bfs_ordered_tables.extend(sorted(hop_tables))

# ── Scalability loop ──────────────────────────────────────────────────────────
print(f"\ndb={args.db_name}  entity_table={entity_table}")
print(f"Total tables: {len(data.node_types)}  "
      f"Total edge types: {len(data.edge_types)}\n")

print(f"{'step':>5}  {'added_table':<30}  {'n_tables':>10}  {'n_edge_types':>13}  "
      f"{'total_params':>13}  {'feat_enc':>10}  {'table_enc':>10}  "
      f"{'relation':>10}  {'fusion':>8}")
print("-" * 115)

for step, added_table in enumerate(bfs_ordered_tables):
    cumulative_tables = bfs_ordered_tables[:step + 1]
    edge_types = subgraph_edge_types(cumulative_tables)

    total, feat_enc, tbl_enc, relation, fusion = count_subgraph_params(
        net, cumulative_tables, edge_types
    )

    print(f"{step:>5}  {added_table:<30}  {len(cumulative_tables):>10}  {len(edge_types):>13}  "
          f"{total:>13,}  {feat_enc:>10,}  {tbl_enc:>10,}  "
          f"{relation:>10,}  {fusion:>8,}")
