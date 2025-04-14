import torch
import math
from torch_geometric.data import HeteroData
from torch_geometric.nn import GraphConv, LayerNorm, Linear, HeteroConv
from torch_geometric.typing import NodeType

from relbench.modeling.nn import HeteroTemporalEncoder
from torch_frame.data.stats import StatType
from model.base import FeatureEncodingModule

from typing import Dict


class HeteroGCN(torch.nn.Module):
    def __init__(
        self,
        data: HeteroData,
        node_to_col_stats: Dict[str, Dict[str, Dict[StatType, torch.nn.Tensor]]],
        channels: int,
        out_channels: int,
        num_layers: int,
        aggr: str,
        edge_aggr: str = 'sum',
        norm: str = 'layer_norm',
        dropout: float = 0.2,
    ):
        super().__init__()

        self.feature_encoder = FeatureEncodingModule(
            data=data,
            node_to_col_stats=node_to_col_stats,
            channels=channels,
        )

        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[
                node_type for node_type in data.node_types if "time" in data[node_type]
            ],
            channels=channels,
        )

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: GraphConv((-1, -1), channels, aggr=aggr)
                for edge_type in data.edge_types
            }, aggr=edge_aggr)
            self.convs.append(conv)

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in data.node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

        self.dropout_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.dropout_dict[node_type] = torch.nn.Dropout(dropout)

        self.lin = Linear(channels, out_channels)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()
        self.lin.reset_parameters()

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType
    ) -> torch.Tensor:
        seed_time = batch[entity_table].seed_time
        B = batch[entity_table].seed_time.size(0)
        x_dict = self.feature_encoder(batch.tf_dict)
        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time.unsqueeze(1)
            # rel_time [B, 1 , channels]

        # [B, attr_num, channels]
        # flatten
        for node_type, x in x_dict.items():
            x_dict[node_type] = x.view(x.size(0), math.prod(x.shape[1:]))

        for _, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            x_dict = conv(x_dict, batch.edge_index_dict)
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}
            x_dict = {key: self.dropout_dict[key](
                x) for key, x in x_dict.items()}
        # [B, out_channels]
        return self.lin(x_dict[entity_table][:B])
