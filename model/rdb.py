import torch
from model.base import HeteroPretrainGNNEncoder
from model.base import NodeRepresentationPart, FeatureEncodingModule
from model.graphsage import HeteroGraphSAGE
from relbench.modeling.nn import HeteroTemporalEncoder
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType
from torch_frame.data.stats import StatType

from typing import Dict


class RDBModel(torch.nn.Module, HeteroPretrainGNNEncoder):
    def __init__(self,
                 data: HeteroData,
                 node_to_col_stats: Dict[str, Dict[str, Dict[StatType, torch.Tensor]]],
                 channels: int,
                 out_channels: int,
                 feat_layer_num: int = 1,
                 feat_norm: str = "layer_norm",
                 head_norm: str = "batch_norm",
                 aggr: str = "sum",
                 graph_layer_num: int = 2,
                 dropout_prob: float = 0.3,
                 ):
        super(RDBModel, self).__init__()

        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[
                node_type for node_type in data.node_types if "time" in data[node_type]
            ],
            channels=channels
        )

        self.node_encoder = NodeRepresentationPart(
            data=data,
            channels=channels,
            num_layers=feat_layer_num,
            normalization=feat_norm,
            dropout_prob=dropout_prob
        )

        self.feature_encoder = FeatureEncodingModule(
            data=data,
            node_to_col_stats=node_to_col_stats,
            channels=channels,
        )

        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=graph_layer_num,
            dropout = dropout_prob,
        )

        self.head = MLP(
            channels,
            out_channels=out_channels,
            norm=head_norm,
            num_layers=1,
        )

        self.out_embedding_dict = torch.nn.ModuleDict(
            {
                node: torch.nn.Sequential(
                    torch.nn.Linear(channels, channels),
                    torch.nn.LayerNorm(channels),
                )
                for node in data.node_types
            }
        )

    def reset_parameters(self):
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        self.feature_encoder.reset_parameters()
        self.node_encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        for emb in self.out_embedding_dict.values():
            torch.nn.init.normal_(emb[0].weight, std=0.1)

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType
    ):
        seed_time = batch[entity_table].seed_time
        x_dict = self.__update_batch_(
            batch,
            entity_table,
        )

        return self.head(x_dict[entity_table][:seed_time.size(0)])

    def get_node_embedding(
        self,
        batch: HeteroData,
        entity_table: NodeType
    ):
        x_dict = self.__update_batch_(
            batch,
            entity_table,
        )

        # add another mapping layer for general node embedding
        x_dict = {
            node: self.out_embedding_dict[node](x)
            for node, x in x_dict.items()
        }

        return x_dict

    def __update_batch_(
        self,
        batch: HeteroData,
        entity_table: NodeType,
    ):
        seed_time = batch[entity_table].seed_time
        x_dict = self.feature_encoder(batch.tf_dict)
        x_dict = self.node_encoder(x_dict)
        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )

        return x_dict




