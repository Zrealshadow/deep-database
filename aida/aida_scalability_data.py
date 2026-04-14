"""
Data Scalability: inference latency vs. number of test instances.
Uses a randomly initialized model (no training needed).
"""

import random
import time
import argparse

import torch
from torch_geometric.loader import NeighborLoader

from utils.util import load_col_types, setup_torch
from utils.resource import get_text_embedder_cfg
from utils.builder import build_pyg_hetero_graph
from utils.data import DatabaseFactory
from utils.sample import get_node_train_table_input_with_sample
from model.aida import construct_default_AIDAXFormer

setup_torch()

parser = argparse.ArgumentParser()
parser.add_argument("--tf_cache_dir", type=str, required=True)
parser.add_argument("--db_name", type=str, required=True)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--device", type=str, default="auto")
parser.add_argument("--num_neighbors", nargs="+", type=int, default=[128, 128])
parser.add_argument("--sample_strategy", type=str, default="last", choices=["last", "uniform"])
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--channels", type=int, default=128)
parser.add_argument("--out_channels", type=int, default=1)
parser.add_argument("--feat_layer_num", type=int, default=1)
parser.add_argument("--feat_nhead", type=int, default=1)
parser.add_argument("--relation_layer_num", type=int, default=2)
parser.add_argument("--relation_aggr", type=str, default="sum")
parser.add_argument("--n_samples_list", nargs="+", type=int,
                    default=[100, 1_000, 10_000, 100_000])
parser.add_argument("--n_repeats", type=int, default=3)
args = parser.parse_args()

# ── Device ────────────────────────────────────────────────────────────────────
if args.device == "auto":
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        selected_gpu = random.randint(0, max(0, num_gpus - 2))
        device = torch.device(f"cuda:{selected_gpu}")
    else:
        device = torch.device("cpu")
else:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

# ── Data ──────────────────────────────────────────────────────────────────────
db = DatabaseFactory.get_db(db_name=args.db_name, with_text_compress=True)
dataset = DatabaseFactory.get_dataset(db_name=args.db_name)
task = DatabaseFactory.get_task(db_name=args.db_name, task_name=args.task, dataset=dataset)

col_type_dict = load_col_types(cache_path=args.tf_cache_dir, file_name="col_type_dict.pkl")
data, col_stats_dict = build_pyg_hetero_graph(
    db, col_type_dict, get_text_embedder_cfg(device="cpu"),
    cache_dir=args.tf_cache_dir, verbose=False,
)

# ── Model (randomly initialized) ──────────────────────────────────────────────
net = construct_default_AIDAXFormer(
    data, col_stats_dict,
    channels=args.channels,
    out_channels=args.out_channels,
    feat_layer_num=args.feat_layer_num,
    feat_nhead=args.feat_nhead,
    relation_layer_num=args.relation_layer_num,
    relation_aggr=args.relation_aggr,
)
net.reset_parameters()
net.to(device)
net.eval()

# ── Test table ────────────────────────────────────────────────────────────────
test_table = task.get_table("test", mask_input_cols=False)
total = len(test_table.df)
print(f"db={args.db_name}  task={args.task}  total_test={total:,}  device={device}")

# ── Build one full-test loader (reused for all n) ─────────────────────────────
_, table_input = get_node_train_table_input_with_sample(
    table=test_table, task=task, sample_ratio=1.0, shuffle=False,
)
loader = NeighborLoader(
    data,
    num_neighbors=args.num_neighbors,
    time_attr="time",
    temporal_strategy=args.sample_strategy,
    input_nodes=table_input.nodes,
    input_time=table_input.time,
    transform=table_input.transform,
    batch_size=args.batch_size,
    shuffle=False,
)


def sync():
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def run_inference(n: int) -> int:
    """Cycle through the loader until n instances have been processed.
    Returns the actual count (may slightly exceed n due to batch granularity)."""
    processed = 0
    while processed < n:
        for batch in loader:
            batch = batch.to(device)
            net(batch, task.entity_table)
            processed += batch[task.entity_table].seed_time.size(0)
            if processed >= n:
                return processed
    return processed


# ── Warm-up ───────────────────────────────────────────────────────────────────
run_inference(min(1000, total))
sync()

# ── Scalability loop ──────────────────────────────────────────────────────────
print(f"\n{'n_req':>12}  {'n_actual':>10}  {'rep':>4}  {'latency(s)':>12}  {'ms/sample':>12}")
print("-" * 58)

for n_req in sorted(args.n_samples_list):
    latencies = []
    n_actual = None

    for rep in range(args.n_repeats):
        sync()
        t0 = time.time()
        n_actual = run_inference(n_req)
        sync()
        latency = time.time() - t0
        latencies.append(latency)
        ms_per = latency / n_actual * 1000
        print(f"{n_req:>12,}  {n_actual:>10,}  {rep:>4}  {latency:>12.3f}  {ms_per:>12.4f}")

    avg = sum(latencies) / len(latencies)
    print(f"{'avg':>12}  {'':>10}  {'':>4}  {avg:>12.3f}  {avg/n_actual*1000:>12.4f}")
    print()
