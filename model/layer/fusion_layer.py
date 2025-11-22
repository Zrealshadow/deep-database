import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Union


from torch_geometric.typing import EdgeType, NodeType, Metadata, Adj
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import scatter


class MultiAggMPConv(MessagePassing):
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
        # return self.lin(x_j)
        return x_j

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
    
    


class FusionLayer(torch.nn.Module):
    """
    Fusion Layer with Self-Attention over Multi-Aggregated Neighbor Features

    For each edge type, computes 4 aggregations (MAX, MIN, SUM, MEAN).
    Uses self-attention over:
    - Target node features (1 token)
    - 4 aggregation tokens per edge type (4 * num_edge_types tokens)

    """

    def __init__(self, node_types, edge_types, in_channels, out_channels, num_heads=1, dropout=0.1):
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads

        assert self.in_channels == self.out_channels, "in_channels must equal out_channels in this FusionLayer"
        # Shared transformations
        self.lin = torch.nn.Linear(in_channels, out_channels)

        # Create conv layers with multi-aggregation
        self.convs = torch.nn.ModuleDict({
            f'{src}__{rel}__{dst}': MultiAggMPConv(self.lin)
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

        self.norm = torch.nn.LayerNorm(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with Xavier uniform initialization."""
        # Initialize linear layers
        # torch.nn.init.xavier_uniform_(self.lin_neighbor.weight)
        # torch.nn.init.zeros_(self.lin_neighbor.bias)

        # torch.nn.init.xavier_uniform_(self.lin_self.weight)
        # torch.nn.init.zeros_(self.lin_self.bias)

        # MultiheadAttention initializes itself, but we can reinitialize if needed
        # The multihead attention module has its own reset_parameters method
        if hasattr(self.multihead_attn, '_reset_parameters'):
            self.multihead_attn._reset_parameters()

        torch.nn.init.xavier_uniform_(self.lin.weight)
        torch.nn.init.zeros_(self.lin.bias)
            

    def forward(self, x_dict, edge_index_dict, entity_table:str, require_attn_weights=False):
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

        # Step 2: Self-attention aggregation only for entity_table
        
        seq_types = []
        attn_weights = None
        node_type = entity_table
        
        # Get self features
        # [num_nodes, out_channels]
        self_feat = x_dict[node_type]

        seq_types.append(f'{node_type}_self')
        tokens_list = [self_feat.unsqueeze(0)]  # [1, num_nodes, out_channels]

        # Self token is never masked (False)
        mask_list = [torch.zeros(self_feat.size(0), dtype=torch.bool, device=self_feat.device)]  # [num_nodes]

        # Add neighbor aggregation tokens if they exist

        if node_type in out_dict and len(out_dict[node_type]) > 0:
            for key in out_dict[node_type].keys():
                # [num_nodes, 4, out_channels]
                aggr = out_dict[node_type][key]

                # Split into 4 separate tokens
                for i in range(4):
                    # [num_nodes, out_channels]
                    aggr_token = aggr[:, i, :]

                    # Check which nodes have zero embeddings (no neighbors)
                    is_zero = torch.all(aggr_token == 0, dim=1)  # [num_nodes]
                    mask_list.append(is_zero)

                    # [1, num_nodes, out_channels]
                    tokens_list.append(aggr_token.unsqueeze(0))
                    seq_types.append(f'{key}_{self.convs[key].aggregators[i]}')
        # Stack all tokens: [seq_len, num_nodes, out_channels]
        # seq_len = 1 (self only) or 1 + 4 * num

        tokens = torch.cat(tokens_list, dim=0)

        # Create attention mask: [num_nodes, seq_len] where True means "ignore this token"
        attn_mask = torch.stack(mask_list, dim=1)  # [num_nodes, seq_len]

        # Self-attention with masking for nodes without neighbors
        # Query, Key, Value are all the same (self-attention)
        # Tokens with no neighbors are masked out
        attn_output, attn_weights = self.multihead_attn(
            query=tokens,  # [seq_len, num_nodes, out_channels]
            key=tokens,
            value=tokens,
            key_padding_mask=attn_mask,  # [num_nodes, seq_len]
            need_weights=require_attn_weights,
            average_attn_weights=require_attn_weights  # Average across attention heads
        )

        # Use only the self token output (first token)
        # attn_output: [seq_len, num_nodes, out_channels]
        self_output = attn_output[0]  # [num_nodes, out_channels]

        # Extract attention weights for self token (how other tokens contribute to self)
        # attn_weights: [num_nodes, seq_len, seq_len] or None
        if attn_weights is not None:
            # Get weights for self token: [num_nodes, seq_len]
            # This shows how much each token (key) contributes to the self token (query)
            self_attn_weights = attn_weights[:, 0, :]  # [num_nodes, seq_len]
        else:
            self_attn_weights = None

        # apply layer Norm
        self_output = self.norm(self_output)
        return self_output, seq_types, self_attn_weights

    
    
    def get_tokens(self, x_dict, edge_index_dict, entity_table:str, require_attn_weights=False):
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

        # Step 2: Self-attention aggregation only for entity_table
        
        seq_types = []
        attn_weights = None
        node_type = entity_table
        
        # Get self features
        # [num_nodes, out_channels]
        self_feat = x_dict[node_type]

        seq_types.append(f'{node_type}_self')
        tokens_list = [self_feat.unsqueeze(0)]  # [1, num_nodes, out_channels]

        # Self token is never masked (False)
        mask_list = [torch.zeros(self_feat.size(0), dtype=torch.bool, device=self_feat.device)]  # [num_nodes]

        # Add neighbor aggregation tokens if they exist

        if node_type in out_dict and len(out_dict[node_type]) > 0:
            for key in out_dict[node_type].keys():
                # [num_nodes, 4, out_channels]
                aggr = out_dict[node_type][key]

                # Split into 4 separate tokens
                for i in range(4):
                    # [num_nodes, out_channels]
                    aggr_token = aggr[:, i, :]

                    # Check which nodes have zero embeddings (no neighbors)
                    is_zero = torch.all(aggr_token == 0, dim=1)  # [num_nodes]
                    mask_list.append(is_zero)

                    # [1, num_nodes, out_channels]
                    tokens_list.append(aggr_token.unsqueeze(0))
                    seq_types.append(f'{key}_{self.convs[key].aggregators[i]}')
        # Stack all tokens: [seq_len, num_nodes, out_channels]
        # seq_len = 1 (self only) or 1 + 4 * num

        tokens = torch.cat(tokens_list, dim=0)

        # Create attention mask: [num_nodes, seq_len] where True means "ignore this token"
        attn_mask = torch.stack(mask_list, dim=1)  # [num_nodes, seq_len]

        # Self-attention with masking for nodes without neighbors
        # Query, Key, Value are all the same (self-attention)
        # Tokens with no neighbors are masked out
        attn_output, attn_weights = self.multihead_attn(
            query=tokens,  # [seq_len, num_nodes, out_channels]
            key=tokens,
            value=tokens,
            key_padding_mask=attn_mask,  # [num_nodes, seq_len]
            need_weights=require_attn_weights,
            average_attn_weights=require_attn_weights  # Average across attention heads
        )

        # Extract attention weights for self token (how other tokens contribute to self)
        if attn_weights is not None:
            self_attn_weights = attn_weights[:, 0, :]  # [num_nodes, seq_len]
        else:
            self_attn_weights = None

        return tokens, seq_types, self_attn_weights