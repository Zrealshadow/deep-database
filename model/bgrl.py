
import torch
import torch.nn.functional as F
import random

from model.base import CompositeModel
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType
from torch_frame.data import TensorFrame

from typing import Dict, Callable, Optional, List
from model.augment import HeteroGraphPermutationTool


class HeteroBGRL(torch.nn.Module):
    """reference to the implementation of BGRL in pyg-ssl
    https://github.com/iDEA-iSAIL-Lab-UIUC/pyg-ssl/blob/main/src/methods/bgrl.py

    Original Paper:LARGE-SCALE REPRESENTATION LEARNING ON GRAPHS VIA BOOTSTRAPPING (ICLR 2022) 
    """

    def __init__(
        self,
        data: HeteroData,
        channels: int,
        student_encoder: CompositeModel,
        teacher_encoder: CompositeModel,
    ):
        super().__init__()

        self.encoder = student_encoder
        self.teacher = teacher_encoder

        self.channels = channels

        self.student_predictor = torch.nn.ModuleDict(
            {
                node_type: torch.nn.Linear(channels, channels)
                for node_type in data.node_types
            }
        )

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType
    ) -> Dict[str, TensorFrame]:

        aug_batch = HeteroGraphPermutationTool.random_permutation(
            batch,
            drop_prob=0.1,
            exclude_node_types=[entity_table],
        )

        B = batch[entity_table].seed_time.size(0)

        x_aug = self.encoder.get_node_embedding(
            aug_batch, entity_table)[entity_table][:B]
        x = self.encoder.get_node_embedding(batch, entity_table)[
            entity_table][:B]
        # [B, channel]

        x_aug = self.student_predictor[entity_table](x_aug)
        x = self.student_predictor[entity_table](x)
        # [B, channel]
        
        with torch.no_grad():
            v = self.teacher.get_node_embedding(batch, entity_table)[
                entity_table][:B]
            v_aug = self.teacher.get_node_embedding(
                aug_batch, entity_table)[entity_table][:B]

        return x, x_aug, v, v_aug

    def discriminate(
        self,
        x: torch.Tensor,
        x_: torch.Tensor,
    ):
        """
        x -> [B, channel]
        """
        x = F.normalize(x, dim=-1, p=2)
        x_ = F.normalize(x_, dim=-1, p=2)
        return 2 - 2 * (x * x_).sum(dim=-1)

    def loss(
        self,
        x: torch.Tensor,
        x_: torch.Tensor,
        v: torch.Tensor,
        v_: torch.Tensor,
    ):
        """
        x, x_ -> [B, channel]
        v, v_ -> [B, channel]
        """
        l1 = self.discriminate(x, v_)
        l2 = self.discriminate(x_, v)
        l = l1 + l2
        return l.mean()
