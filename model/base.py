# declare the model
from typing import Any, Dict, List, Optional

import math
import torch
import torch_frame
from torch import Tensor
from torch.nn import Embedding, ModuleDict
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_frame.nn.models.resnet import FCResidualBlock
from torch_geometric.typing import NodeType
from torch_frame.nn.encoder import StypeWiseFeatureEncoder
# from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder

from model.graphsage import HeteroGraphSAGE


class FeatureEncodingPart(torch.nn.Module):
    '''convert the raw feature to the feature embedding.
    '''

    def __init__(
        self,
        data: HeteroData,
        node_to_col_stats: Dict[str, Dict[str, Dict[StatType, Tensor]]],
        channels: int
    ):
        super().__init__()
        self.encoders = torch.nn.ModuleDict()
        # node_type : StypeWiseFeatureEncoder

        node_to_col_names_dict = {
            node_type: data[node_type].tf.col_names_dict
            for node_type in data.node_types
        }
        # node_type:  {stype: [col_name]}

        default_stype_encoder_cls_kwargs: Dict[torch_frame.stype, Any] = {
            torch_frame.categorical: (torch_frame.nn.EmbeddingEncoder, {}),
            torch_frame.numerical: (torch_frame.nn.LinearEncoder, {}),
            torch_frame.multicategorical: (
                torch_frame.nn.MultiCategoricalEmbeddingEncoder,
                {},
            ),
            torch_frame.embedding: (torch_frame.nn.LinearEmbeddingEncoder, {}),
            torch_frame.timestamp: (torch_frame.nn.TimestampEncoder, {}),
        }

        for node_type in node_to_col_names_dict.keys():
            stype_encoder_dict = {
                stype: default_stype_encoder_cls_kwargs[stype][0](
                    **default_stype_encoder_cls_kwargs[stype][1]
                )
                for stype in node_to_col_names_dict[node_type].keys()
            }
            self.encoders.update({node_type: StypeWiseFeatureEncoder(
                out_channels=channels,
                col_stats=node_to_col_stats[node_type],
                col_names_dict=node_to_col_names_dict[node_type],
                stype_encoder_dict=stype_encoder_dict
            )})

    def reset_parameters(self):
        for encoder in self.encoders.values():
            encoder.reset_parameters()

    def forward(
        self,
        tf_dict: Dict[NodeType, torch_frame.TensorFrame],
    ) -> Dict[NodeType, Tensor]:
        x_dict = {}
        for node_type, tf in tf_dict.items():
            x, _ = self.encoders[node_type](tf)
            x_dict[node_type] = x
        return x_dict


class NodeRepresentationPart(torch.nn.Module):
    '''generate node/tuple representation from feature encoding.
    '''

    def __init__(
        self,
        data: HeteroData,
        channels: int,
        num_layers: int,
        normalization: Optional[str] = "layer_norm",
        dropout_prob: float = 0.0
    ):
        super().__init__()

        self.mappers = torch.nn.ModuleDict()

        node_to_col_names_dict = {
            node_type: data[node_type].tf.col_names_dict
            for node_type in data.node_types
        }
        # node_type:  {stype: [col_name]}

        for node_type, type_to_col_names in node_to_col_names_dict.items():
            col_cnt = 0
            for cols in type_to_col_names.values():
                col_cnt += len(cols)
            in_channels = col_cnt * channels
            backbone = torch.nn.Sequential(*[
                FCResidualBlock(
                    in_channels if i == 0 else channels,
                    channels,
                    normalization=normalization,
                    dropout_prob=dropout_prob
                )
                for i in range(num_layers)],
                torch.nn.LayerNorm(channels),
                torch.nn.ReLU(),
                torch.nn.Linear(channels, channels)
            )
            self.mappers.update(
                {
                    node_type: backbone
                }
            )

    def reset_parameters(self):
        for mapper in self.mappers.values():
            for layer in mapper:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()

    def forward(self,
                x_dict: Dict[NodeType, Tensor]
                ) -> Dict[NodeType, Tensor]:
        out_dict = {}
        for node_type, x in x_dict.items():
            # Flattening the encoder output
            x = x.view(x.size(0), math.prod(x.shape[1:]))
            out_dict[node_type] = self.mappers[node_type](x)
        return out_dict


class CompositeModel(torch.nn.Module):

    def __init__(
        self,
        data: HeteroData,
        channels: int,
        out_channels: int,
        dropout: float,
        aggr: str,
        norm: str,
        num_layer: int,
        feature_encoder: torch.nn.Module,
        node_encoder: torch.nn.Module,
        temporal_encoder: torch.nn.Module,
        shallow_list: List[NodeType] = [],
        id_awareness: bool = False,
    ):
        super().__init__()
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layer,
            dropout=dropout
        )

        self.head = MLP(
            channels,
            out_channels=out_channels,
            norm=norm,
            num_layers=1
        )

        self.feature_encoder = feature_encoder
        self.node_encoder = node_encoder
        self.temporal_encoder = temporal_encoder

        self.embedding_dict = ModuleDict(
            {
                node: torch.nn.Embedding(data.num_nodes_dict[node], channels)
                for node in shallow_list
            }
        )

        self.id_awareness_emb = None
        if id_awareness:
            self.id_awareness_emb = torch.nn.Embedding(1, channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        self.feature_encoder.reset_parameters()
        self.node_encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()

        for emb in self.embedding_dict.values():
            torch.nn.init.normal_(emb.weight, std=0.1)

        if self.id_awareness_emb is not None:
            torch.nn.init.normal_(self.id_awareness_emb.weight, std=0.1)

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType
    ) -> Tensor:
        seed_time = batch[entity_table].seed_time
        x_dict = self.feature_encoder(batch.tf_dict)
        x_dict = self.node_encoder(x_dict)
        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + \
                embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )

        return self.head(x_dict[entity_table][: seed_time.size(0)])
