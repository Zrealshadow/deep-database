
from torch_geometric.data import HeteroData
from relbench.modeling.nn import HeteroTemporalEncoder

from torch_geometric.nn import LayerNorm
from torch_geometric.typing import EdgeType, NodeType
import torch

from typing import Dict, Any, List

from .graphsage import HeteroGraphSAGE
from .layer.relationConv import AttentionHeteroConv
from .base import default_stype_encoder_cls_kwargs, construct_stype_encoder_dict
from .encoder import build_encoder

import torch_frame
from torch_frame import stype
from torch_frame.data.stats import StatType
from torch_frame.nn import StypeWiseFeatureEncoder


class AIDABaseFeatureEncoder(torch.nn.Module):
    """Base feature encoder for AIDA.
    Convert the raw features of multiple tables into embeddings based on corresponding type.
    Specifically, each table has its own StypeWiseFeatureEncoder.
    """

    def __init__(self,
                 channels: int,
                 tables_col_names_dict: Dict[str, Dict[stype, List[str]]],
                 tables_col_stats: Dict[str, Dict[str, stype]],
                 stype_encoder_cls_kwargs: Dict[stype, Any] = None,
                 ):

        super().__init__()

        assert set(tables_col_stats.keys()) == set(tables_col_names_dict.keys()), \
            "tables_col_stats and tables_col_names_dict must have the same tables."

        self.feature_encoders = torch.nn.ModuleDict()

        stype_encoder_cls_kwargs = stype_encoder_cls_kwargs if \
            stype_encoder_cls_kwargs else default_stype_encoder_cls_kwargs

        stype_encoder_dict = construct_stype_encoder_dict(
            stype_encoder_cls_kwargs)

        for table, col_stype_dict in tables_col_names_dict.items():
            col_stats = tables_col_stats[table]

            stype_encoder_dict = construct_stype_encoder_dict(
                stype_encoder_cls_kwargs)

            self.feature_encoders.update({table: StypeWiseFeatureEncoder(
                out_channels=channels,
                col_stats=col_stats,
                col_names_dict=col_stype_dict,
                stype_encoder_dict=stype_encoder_dict,
            )
            })

    def reset_parameters(self):
        for encoder in self.feature_encoders.values():
            encoder.reset_parameters()

    def forward(self,
                table_batch_dict: Dict[str, torch_frame.TensorFrame],
                ) -> Dict[str, torch.Tensor]:

        x_dict = {}
        # for each one table, the size of output is [B,C,channels
        # C the number of columns in this table
        for table, tensor_frame in table_batch_dict.items():
            x_dict[table], _ = self.feature_encoders[table](tensor_frame)
        return x_dict


class AIDASharedTableEncoder(torch.nn.Module):

    def __init__(self,
                 tables: List[str],
                 feat_channels: int,
                 channels: int,
                 num_layers: int = 1,
                 nhead: int = 4,
                 dropout_prob: float = 0.1,
                 activation: str = "gelu",
                 ):
        super().__init__()

        self.table_cls_embeddings = torch.nn.ParameterDict()

        for table in tables:
            self.table_cls_embeddings[table] = torch.nn.Parameter(
                torch.randn(feat_channels)
            )

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=feat_channels,
            nhead=nhead,
            dim_feedforward=channels,
            dropout=dropout_prob,
            activation=activation,
            batch_first=True,
        )

        encoder_norm = torch.nn.LayerNorm(feat_channels)

        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=encoder_norm,
        )

        self.norm = torch.nn.LayerNorm(feat_channels)

    def reset_parameters(self):
        for emb in self.table_cls_embeddings.values():
            torch.nn.init.normal_(emb, std=0.01)

        for p in self.transformer.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def calculate_table_embedding(self,
                                  table_name: str,
                                  table_batch: torch.Tensor):
        B, _, _ = table_batch.shape
        x_cls = self.table_cls_embeddings[table_name]
        x_cls = x_cls.repeat(B, 1, 1)
        x_concate = torch.cat([x_cls, table_batch], dim=1)
        x_concate = self.transformer(x_concate)
        x_mean = torch.mean(x_concate, dim=1)
        x = self.norm(x_mean)
        return x

    def forward(self,
                table_batch_dict: Dict[str, torch.Tensor],
                ) -> Dict[str, torch.Tensor]:
        """
        table_batch_dict: Dict of table name to tensor of shape
            (B, C, feat_channels)
        """
        x_dict = {}
        x_cls_dict = {}

        for table, x in table_batch_dict.items():
            B, _, _ = x.shape
            x_cls = self.table_cls_embeddings[table]
            # reshape cls to (B)
            x_cls = x_cls.repeat(B, 1, 1)

            # pad corresponding cls token to the each table batch
            x_concate = torch.cat([x_cls, x], dim=1)
            x_concate = self.transformer(x_concate)

            # x_cls, x
            # x_cls, x = x_concate[:, 0, :], x_concate[:, 1:, :]
            x_cls = x_concate[:, 0, :]
            x = torch.mean(x_concate, dim=1)

            # concate the cls token embedding with the mean pooling of other attributes
            x = self.norm(x)
            # x = self.norm(torch.cat([x_cls, x_mean], dim=1))
            # x (B, 2*feat_channels)
            x_dict[table], x_cls_dict[table] = x, x_cls

        return x_cls_dict, x_dict


class AIDARelationModule(torch.nn.Module):
    """
        Relation module based on customized RelationConv
        GraphSAGE-style
    """

    def __init__(
        self,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        channels: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()

        for _ in range(num_layers):
            conv = AttentionHeteroConv(
                edge_types=edge_types,
                in_channels=channels,
                out_channels=channels,
                dropout=dropout,
            )
            self.convs.append(conv)

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm = LayerNorm(channels, mode="node")
            self.norms.append(norm)

        self.dropouts = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.dropouts.append(torch.nn.Dropout(dropout))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()

    def forward(
        self,
        x_dict: Dict[NodeType, torch.Tensor],
        edge_index_dict: Dict[EdgeType, torch.Tensor],
        num_sampled_nodes_dict: Dict[NodeType, List[int]] = None,
        num_sampled_edges_dict: Dict[EdgeType, List[int]] = None,
    ) -> Dict[NodeType, torch.Tensor]:
        for _, (conv, norm, drop) in enumerate(zip(self.convs, self.norms, self.dropouts)):
            x_dict, _, _ = conv(
                x_dict,
                edge_index_dict
            )
            x_dict = {key: norm(x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}
            x_dict = {key: drop(x) for key, x in x_dict.items()}
        return x_dict


class AIDABasicFormer(torch.nn.Module):
    """
        A basic model for relational data
        - AIDABaseFeatureEncoder:
        - AIDASharedTableEncoder:
        - AIDARelationModule:
        - AIDAFusionModule (Not implemented yet)
    """

    def __init__(
            self,
            data: HeteroData,
            node_to_col_stats: Dict[str, Dict[str, StatType]],
            channels: int,
            out_channels: int,
            feat_layer_num: int = 2,
            graph_layer_num: int = 2,
            feat_nhead: int = 1,
            graph_nhead: int = 1,
            dropout_prob: float = 0.1):
        super().__init__()

        table_col_names_dict = {
            table: data[table].tf.col_names_dict
            for table in data.node_types
        }
        self.feat_encoder = AIDABaseFeatureEncoder(
            channels=channels,
            tables_col_names_dict=table_col_names_dict,
            tables_col_stats=node_to_col_stats,
        )

        self.basic_encoder = AIDASharedTableEncoder(
            tables=data.node_types,
            feat_channels=channels,
            channels=channels,
            num_layers=feat_layer_num,
            dropout_prob=dropout_prob,
            nhead=feat_nhead,
        )

        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[
                node_type for node_type in data.node_types if "time" in data[node_type]
            ],
            channels=2*channels
        )

        self.relation_module = AIDARelationModule(
            data.node_types,
            data.edge_types,
            channels=2 * channels,
            num_layers=graph_layer_num,
            dropout=dropout_prob,
        )

        self.head = torch.nn.Linear(2*channels, out_channels)

    def reset_parameters(self):
        self.temporal_encoder.reset_parameters()
        self.feat_encoder.reset_parameters()
        self.basic_encoder.reset_parameters()
        self.relation_module.reset_parameters()
        torch.nn.init.xavier_uniform_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType,
    ):
        seed_time = batch[entity_table].seed_time
        x_dict = self.feat_encoder(batch.tf_dict)
        # {table: (B, feat_channels)}
        x_dict = self.basic_encoder(x_dict)[1]

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        x_dict = self.relation_module(
            x_dict,
            batch.edge_index_dict,
        )

        return self.head(x_dict[entity_table][:seed_time.size(0)])


class AIDATableEncoder(torch.nn.Module):
    """
        if specific_table_encoder is given, use it to encode the corresponding table.
        Otherwise, use shared table encoder. If can be seem as a general version of AIDASharedTableEncoder.
    """

    def __init__(
            self,
            tables: List[str],
            feat_channels: int,
            channels: int,
            num_layers: int = 1,
            nhead: int = 1,
            dropout_prob: float = 0.1,
            activation: str = "gelu",
            specific_table_encoder: Dict[str, torch.nn.Module] = None,
    ):
        super(AIDATableEncoder, self).__init__()

        self.shared_table_encoder = AIDASharedTableEncoder(
            tables=tables,
            feat_channels=feat_channels,
            channels=channels,
            num_layers=num_layers,
            nhead=nhead,
            dropout_prob=dropout_prob,
            activation=activation,
        )
        self.channels = feat_channels
        self.table_encoder_dict = torch.nn.ModuleDict(specific_table_encoder)
        # if is not none, check the channels are matched
        if specific_table_encoder is not None:
            for table, encoder in specific_table_encoder.items():
                # assert the encoder has reset_parameters method
                assert hasattr(
                    encoder, 'reset_parameters'), "The specific table encoder must have reset_parameters method."
                assert hasattr(
                    encoder, 'channels'), "The specific table encoder must have attribute 'channels'."
                assert encoder.channels == feat_channels, "The specific table encoder channels must match feat_channels."

    def forward(self, x_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for table, x in x_dict.items():
            if table in self.table_encoder_dict:
                x_dict[table] = self.table_encoder_dict[table](x)
            else:
                x_dict[table] = self.shared_table_encoder.calculate_table_embedding(
                    table, x
                )
        return x_dict

    def reset_parameters(self):
        self.shared_table_encoder.reset_parameters()
        for encoder in self.table_encoder_dict.values():
            encoder.reset_parameters()


class AIDAXFormer(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        database_feature_encoder,
        database_table_encoder,
        temporal_encoder,
        relation_module,
        fusion_module,
    ):
        super(AIDAXFormer, self).__init__()
        self.feat_encoder = database_feature_encoder
        self.table_encoder = database_table_encoder
        self.temporal_encoder = temporal_encoder

        self.relation_module = relation_module
        self.fusion_mododule = fusion_module

        self.head = torch.nn.Linear(channels, out_channels)

    def reset_parameters(self):
        self.feat_encoder.reset_parameters()
        self.table_encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.relation_module.reset_parameters()
        # self.fusion_mododule.reset_parameters()
        torch.nn.init.xavier_uniform_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType,
    ):
        seed_time = batch[entity_table].seed_time
        x_dict = self.feat_encoder(batch.tf_dict)
        # {table: (B, feat_channels)}
        x_dict = self.table_encoder(x_dict)

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        x_dict = self.relation_module(
            x_dict,
            batch.edge_index_dict,
        )

        return self.head(x_dict[entity_table][:seed_time.size(0)])

# --------------- Helper functions---------------- #
def construct_default_AIDAXFormer(
    data: HeteroData,
    node_to_col_stats: Dict[str, Dict[str, StatType]],
    channels: int,
    out_channels: int,
    feat_layer_num: int = 2,
    graph_layer_num: int = 2,
    feat_nhead: int = 4,
    graph_nhead: int = 1,
    dropout_prob: float = 0.1,
    specific_table_encoder: Dict[str, str] = None,
) -> AIDAXFormer:

    table_col_names_dict = {
        table: data[table].tf.col_names_dict
        for table in data.node_types
    }

    database_feature_encoder = AIDABaseFeatureEncoder(
        channels=channels,
        tables_col_names_dict=table_col_names_dict,
        tables_col_stats=node_to_col_stats,
    )

    if specific_table_encoder is not None:
        # convert the table encoder to actuat torch nn
        specific_table_encoder = {
            table: build_encoder(
                encoder_type, channels, num_layers=feat_layer_num, dropout_prob=dropout_prob)
            for table, encoder_type in specific_table_encoder.items()
        }

    database_table_encoder = AIDATableEncoder(
        tables=data.node_types,
        feat_channels=channels,
        channels=channels,
        num_layers=feat_layer_num,
        nhead=feat_nhead,
        dropout_prob=dropout_prob,
        specific_table_encoder=specific_table_encoder,
    )

    temporal_encoder = HeteroTemporalEncoder(
        node_types=[
            node_type for node_type in data.node_types if "time" in data[node_type]
        ],
        channels=channels
    )

    relation_module = HeteroGraphSAGE(
        data.node_types,
        edge_types=data.edge_types,
        channels=channels,
        aggr="sum",
        num_layers=graph_layer_num,
        dropout=dropout_prob,
    )

    fusion_module = None  # Not implemented yet

    model = AIDAXFormer(
        channels=channels,
        out_channels=out_channels,
        database_feature_encoder=database_feature_encoder,
        database_table_encoder=database_table_encoder,
        temporal_encoder=temporal_encoder,
        relation_module=relation_module,
        fusion_module=fusion_module,
    )

    return model


if __name__ == '__main__':
    pass
