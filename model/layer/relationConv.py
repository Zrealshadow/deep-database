import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Union


from torch_geometric.typing import EdgeType, NodeType, Metadata, Adj
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import scatter


class SharedMessagePassingConv(MessagePassing):
    """
    Multi-aggregation message passing with MAX, MIN, SUM, MEAN.

    Returns:
        [num_nodes, 4, channels] where dim=1 contains [max, min, sum, mean] aggregations
    """

    def __init__(self, shared_lin):
        super().__init__()  # Use custom aggregation
        self.lin = shared_lin  # Shared across all edge types
        self.aggregators = ['max', 'min', 'sum', 'mean']

    def forward(self, x, edge_index):
        if isinstance(x, tuple):
            x_src, x_dst = x
            size = (x_src.size(0), x_dst.size(0))
        else:
            x_src = x_dst = x
            size = None

        # Returns [num_nodes, 4, channels]
        return self.propagate(edge_index, x=x_src, size=size)

    def message(self, x_j):
        """Transform neighbor features."""
        return self.lin(x_j)

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        """
        Multi-aggregation: MAX, MIN, SUM, MEAN

        Args:
            inputs: messages [num_edges, channels]
            index: target node indices [num_edges]
            ptr: Optional CSR pointer (not used)
            dim_size: number of target nodes

        Returns:
            Stacked aggregations [num_nodes, 4, channels]
        """
        outputs = []

        for aggr in self.aggregators:
            out = scatter(
                inputs,
                index,
                dim=0,
                dim_size=dim_size,
                reduce=aggr
            )
            outputs.append(out)

        # Stack: [num_nodes, 4, channels]
        return torch.stack(outputs, dim=1)

    def update(self, aggr_out):
        """Just return aggregated neighbors."""
        return aggr_out


class AttentionHeteroConv(torch.nn.Module):
    """
    Heterogeneous graph conv with multi-aggregation and self-attention.

    For each edge type, computes 4 aggregations (MAX, MIN, SUM, MEAN).
    Uses self-attention over:
    - Target node features (1 token)
    - 4 aggregation tokens per edge type (4 * num_edge_types tokens)

    Zero aggregations are masked out in attention. 
    """

    def __init__(self, edge_types, in_channels, out_channels, num_heads=1, dropout=0.1):
        super().__init__()
        self.edge_types = edge_types
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads

        # Shared transformations
        self.lin_neighbor = torch.nn.Linear(in_channels, out_channels)
        self.lin_self = torch.nn.Linear(in_channels, out_channels)

        # Create conv layers with multi-aggregation
        self.convs = torch.nn.ModuleDict({
            f'{src}__{rel}__{dst}': SharedMessagePassingConv(self.lin_neighbor)
            for src, rel, dst in edge_types
        })

        # Multi-head self-attention
        # Each token has out_channels dimensions
        self.multihead_attn = torch.nn.MultiheadAttention(
            embed_dim=out_channels,
            num_heads=num_heads,
            batch_first=False,  # expects [seq_len, batch, embed_dim]
            dropout=dropout
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with Xavier uniform initialization."""
        # Initialize linear layers
        torch.nn.init.xavier_uniform_(self.lin_neighbor.weight)
        torch.nn.init.zeros_(self.lin_neighbor.bias)

        torch.nn.init.xavier_uniform_(self.lin_self.weight)
        torch.nn.init.zeros_(self.lin_self.bias)

        # MultiheadAttention initializes itself, but we can reinitialize if needed
        # The multihead attention module has its own reset_parameters method
        if hasattr(self.multihead_attn, '_reset_parameters'):
            self.multihead_attn._reset_parameters()

    def forward(self, x_dict, edge_index_dict, require_attn_weights=False):
        out_dict = {}

        # Step 1: Per edge-type message passing (multi-aggregation)
        for edge_type, edge_index in edge_index_dict.items():
            src, rel, dst = edge_type
            key = f'{src}__{rel}__{dst}'

            x_src = x_dict[src]
            x_dst = x_dict[dst] if src != dst else x_src

            # Returns [num_nodes, 4, out_channels] (4 aggregations)
            neighbor_aggr = self.convs[key]((x_src, x_dst), edge_index)

            if dst not in out_dict:
                out_dict[dst] = {}
            out_dict[dst][key] = neighbor_aggr

        # Step 2: Self-attention aggregation
        result = {}
        node_type_seq_types = {}
        node_type_attn_weights = {}
        for node_type in x_dict.keys():
            # Get self features
            # [num_nodes, out_channels]
            self_feat = self.lin_self(x_dict[node_type])
            num_nodes = self_feat.size(0)

            # Always start with self token
            node_type_seq_types[node_type] = [f'{node_type}_self']
            tokens_list = [self_feat.unsqueeze(0)]  # [1, num_nodes, out_channels]

            # Add neighbor aggregation tokens if they exist
            if node_type in out_dict and len(out_dict[node_type]) > 0:
                for key in out_dict[node_type].keys():
                    # [num_nodes, 4, out_channels]
                    aggr = out_dict[node_type][key]
                    # Split into 4 separate tokens
                    for i in range(4):
                        # [1, num_nodes, out_channels]
                        tokens_list.append(aggr[:, i, :].unsqueeze(0))
                        node_type_seq_types[node_type].append(f'{key}_{self.convs[key].aggregators[i]}')

            # Stack all tokens: [seq_len, num_nodes, out_channels]
            # seq_len = 1 (self only) or 1 + 4 * num_edge_types (with neighbors)
            tokens = torch.cat(tokens_list, dim=0)

            # Self-attention (always applied, even for isolated nodes)
            # Query, Key, Value are all the same (self-attention)
            # No masking - zero tokens are meaningful features (indicate no edges)
            attn_output, attn_weights = self.multihead_attn(
                query=tokens,  # [seq_len, num_nodes, out_channels]
                key=tokens,
                value=tokens,
                need_weights=require_attn_weights,
                average_attn_weights=require_attn_weights  # Average across attention heads
            )

            # Store attention weights for each node type
            node_type_attn_weights[node_type] = attn_weights

            # Take the output for the first token (self token) + residual
            # This represents the updated node features
            # mean pool the attn_output over all heads
            attn_output = attn_output.mean(dim=0) # [num_nodes, out_channels]
            out = self_feat + attn_output  # [num_nodes, out_channels]

            result[node_type] = out

        return result, node_type_seq_types, node_type_attn_weights
