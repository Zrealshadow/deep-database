import torch
import math
from torch_geometric.data import HeteroData
from torch_geometric.nn import LayerNorm, Linear, HeteroConv, HGTConv, BatchNorm
from torch_geometric.typing import NodeType

from relbench.modeling.nn import HeteroTemporalEncoder
from torch_frame.data.stats import StatType
from model.base import FeatureEncodingModule
from typing import Dict


class HGT(torch.nn.Module):

    def __init__(
        self,
        data: HeteroData,
        node_to_col_stats: Dict[str, Dict[str, Dict[StatType, torch.Tensor]]],
        channels: int,
        out_channels: int,
        num_layers: int,
        aggr: str,
        edge_aggr: str = 'sum',
        norm: str = 'layer_norm',
        dropout: float = 0.2,
        heads: int = 4,
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

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            col_cnt = data[node_type].tf.num_cols
            in_channels = col_cnt * channels
            self.lin_dict[node_type] = Linear(
                in_channels, channels
            )

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(
                channels,
                out_channels=channels,
                metadata=data.metadata(),
                heads=heads,
            )
            self.convs.append(conv)

            norm_dict = torch.nn.ModuleDict()
            for node_type in data.node_types:
                if norm == "layer_norm":
                    norm_dict[node_type] = LayerNorm(channels, mode="node")
                elif norm == "batch_norm":
                    norm_dict[node_type] = BatchNorm(channels)
                else:
                    raise ValueError(f"Unknown normalization type: {norm}")
            self.norms.append(norm_dict)

        self.dropout_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.dropout_dict[node_type] = torch.nn.Dropout(dropout)

        self.lin = Linear(channels, out_channels)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()
        self.lin.reset_parameters()

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType,
    ) -> torch.Tensor:
        seed_time = batch[entity_table].seed_time
        B = seed_time.size(0)
        x_dict = self.feature_encoder(batch.tf_dict)
        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, x in x_dict.items():
            x_dict[node_type] = x.view(x.size(0), math.prod(x.shape[1:]))
            x_dict[node_type] = self.lin_dict[node_type](x_dict[node_type])
        # [B, attr_num, channels] -> [B, attr_num * channels]
        # [B, channels]

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for _, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            x_dict = conv(x_dict, batch.edge_index_dict)
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}
            x_dict = {key: self.dropout_dict[key](
                x) for key, x in x_dict.items()}

        return self.lin(x_dict[entity_table][:B])
