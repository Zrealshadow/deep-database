import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import scatter


class MultiAggregatorConv(MessagePassing):
    """
    Multi-aggregation message passing without transformation.
    Aggregates with Mean, Max, Min, Sum.

    Returns:
        [num_nodes, 4, in_channels] where dim=1 contains [mean, max, min, sum]
    """

    def __init__(self):
        super().__init__()
        self.aggregators = ['mean', 'max', 'min', 'sum']

    def forward(self, x, edge_index):
        """
        Args:
            x: source node features [num_src_nodes, in_channels]
            edge_index: [2, num_edges]

        Returns:
            [num_dst_nodes, 4, in_channels]
        """
        if isinstance(x, tuple):
            x_src, x_dst = x
            size = (x_src.size(0), x_dst.size(0))
        else:
            x_src = x_dst = x
            size = None

        # Returns [num_nodes, 4, channels]
        return self.propagate(edge_index, x=x_src, size=size)

    def message(self, x_j):
        """No transformation, just pass through neighbor features."""
        return x_j

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        """
        Multi-aggregation: Mean, Max, Min, Sum

        Args:
            inputs: messages [num_edges, channels]
            index: target node indices [num_edges]
            dim_size: number of target nodes

        Returns:
            [num_nodes, 4, channels]
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
        """Return aggregated output."""
        return aggr_out


class DynamicRelationConv(torch.nn.Module):
    """
    Dynamic Relation Modeling Layer following the paper formulation.

    Architecture:
    1. Intra-relation aggregation: s_{v,r}^(l) = Concat[Mean, Max, Min, Sum](h_u^(l): u in N_r(v))
    2. Relation-aware transformation: h_{v,r}^(l+1) = W_r^1 * h_v^(l) + W_r^2 * s_{v,r}^(l)
    3. Cross-relation aggregation: h_v^(l+1) = LayerNorm(sigma(1/|R| * sum(h_{v,r}^(l+1))))

    Args:
        edge_types: List of (src, rel, dst) tuples defining the heterogeneous graph schema
        in_channels: Input feature dimension
        out_channels: Output feature dimension
    """

    def __init__(self, node_types, edge_types, in_channels, out_channels):
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Build relation type set per destination node type
        # rel_types_per_node[dst_type] = [edge_type_key1, edge_type_key2, ...]
        self.rel_types_per_node = {}
        for src, rel, dst in edge_types:
            edge_type_key = f'{src}__{rel}__{dst}'
            if dst not in self.rel_types_per_node:
                self.rel_types_per_node[dst] = []
            self.rel_types_per_node[dst].append(edge_type_key)

        # Step 1: Multi-aggregator convolutions (no transformation)
        self.convs = torch.nn.ModuleDict({
            f'{src}__{rel}__{dst}': MultiAggregatorConv()
            for src, rel, dst in edge_types
        })

        # Step 2: Relation-specific transformation parameters
        # W_r^1: [in_channels, out_channels] for self features
        # W_r^2: [4*in_channels, out_channels] for concatenated aggregations
        self.W_r1 = torch.nn.ModuleDict()
        self.W_r2 = torch.nn.ModuleDict()

        for src, rel, dst in edge_types:
            edge_type_key = f'{src}__{rel}__{dst}'
            self.W_r1[edge_type_key] = torch.nn.Linear(in_channels, out_channels, bias=False)
            # 4*in_channels because we concatenate [mean, max, min, sum]
            self.W_r2[edge_type_key] = torch.nn.Linear(4 * in_channels, out_channels, bias=False)

        # Step 3: Cross-relation aggregation normalization
        self.norm = torch.nn.LayerNorm(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with Xavier uniform."""
        for lin in self.W_r1.values():
            torch.nn.init.xavier_uniform_(lin.weight)
        for lin in self.W_r2.values():
            torch.nn.init.xavier_uniform_(lin.weight)

    def forward(self, x_dict, edge_index_dict):
        """
        Args:
            x_dict: {node_type: [num_nodes, in_channels]} - node features per type
            edge_index_dict: {edge_type: [2, num_edges]} - edges per relation type

        Returns:
            {node_type: [num_nodes, out_channels]} - updated node features
        """
        # Step 1: Intra-relation aggregation
        # For each edge type, aggregate neighbor features
        # s_{v,r}^(l) = Concat[Mean, Max, Min, Sum](h_u^(l): u in N_r(v))

        out_dict = {}  # Stores per-relation aggregations

        for edge_type, edge_index in edge_index_dict.items():
            src, rel, dst = edge_type
            edge_type_key = f'{src}__{rel}__{dst}'

            # Get source node features
            x_src = x_dict[src]

            # Multi-aggregation: [num_dst_nodes, 4, in_channels]
            aggr_out = self.convs[edge_type_key](x_src, edge_index)

            # Store by destination node type and edge type
            if dst not in out_dict:
                out_dict[dst] = {}
            out_dict[dst][edge_type_key] = aggr_out

        # Step 2 & 3: Relation-aware transformation + Cross-relation aggregation
        result = {}

        for node_type in x_dict.keys():
            x_v = x_dict[node_type]  # [num_nodes, in_channels]
            num_nodes = x_v.size(0)

            # Check if this node type has incoming edges
            if node_type not in self.rel_types_per_node:
                # No incoming edge types defined for this node type
                # Apply identity or zero transformation
                result[node_type] = torch.zeros(
                    num_nodes, self.out_channels,
                    device=x_v.device, dtype=x_v.dtype
                )
                continue

            # Collect h_{v,r}^(l+1) for all relation types targeting this node
            h_vr_list = []

            for edge_type_key in self.rel_types_per_node[node_type]:
                # Step 2: Relation-aware transformation
                # h_{v,r}^(l+1) = W_r^1 * h_v^(l) + W_r^2 * s_{v,r}^(l)

                # Self transformation: W_r^1 * h_v^(l)
                h_self = self.W_r1[edge_type_key](x_v)  # [num_nodes, out_channels]

                # Neighbor aggregation transformation: W_r^2 * s_{v,r}^(l)
                if node_type in out_dict and edge_type_key in out_dict[node_type]:
                    # s_{v,r}^(l): [num_nodes, 4, in_channels]
                    s_vr = out_dict[node_type][edge_type_key]
                    # Flatten to [num_nodes, 4*in_channels] for concat
                    s_vr_flat = s_vr.view(num_nodes, -1)
                    # Transform: W_r^2 * s_{v,r}
                    h_neighbor = self.W_r2[edge_type_key](s_vr_flat)  # [num_nodes, out_channels]
                else:
                    # No edges of this type - zero aggregation is meaningful
                    h_neighbor = torch.zeros_like(h_self)

                # Combine self and neighbor information
                h_vr = h_self + h_neighbor  # [num_nodes, out_channels]
                h_vr_list.append(h_vr)

            # Step 3: Cross-relation aggregation
            # h_v^(l+1) = LayerNorm(sigma(1/|R| * sum_{r in R} h_{v,r}^(l+1)))
            if len(h_vr_list) > 0:
                # Average across all relation types
                h_v = torch.stack(h_vr_list, dim=0).mean(dim=0)  # [num_nodes, out_channels]

                # Apply non-linearity
                h_v = F.relu(h_v)

                # Apply layer normalization
                h_v = self.norm(h_v)

                result[node_type] = h_v
            else:
                # No relations processed (shouldn't happen if rel_types_per_node is correct)
                result[node_type] = torch.zeros(
                    num_nodes, self.out_channels,
                    device=x_v.device, dtype=x_v.dtype
                )

        return result
