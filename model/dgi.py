import torch
from model.base import CompositeModel
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType
from torch_frame.data import TensorFrame

from typing import Dict, Callable, Optional


class HeteroDeepGraphInfomax(torch.nn.Module):
    """refer to the implementation of DGI in pyg
    """
    def __init__(
        self,
        data: HeteroData,
        channel: int,
        encoder: CompositeModel,
        summary: Optional[Callable] = None,
        corruption: Optional[Callable] = None,
    ):
        super().__init__()
        self.channel = channel
        self.encoder = encoder
        self.summary = summary
        self.corruption = corruption

        self.mapper_dict = torch.nn.ModuleDict(
            {
                node: torch.nn.Linear(channel, channel, bias=False)
                for node in data.node_types
            }
        )

    def reset_parameter(self):
        """Reset parameters of the model.
        """
        for node in self.mapper_dict:
            torch.nn.init.xavier_uniform_(self.mapper_dict[node].weight)
        self.encoder.reset_parameters()

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType
    ):
        x_dict_pos = self.encoder.get_node_embedding(batch, entity_table)
        self._corruption(batch)
        x_dict_neg = self.encoder.get_node_embedding(batch, entity_table)

        # compute the summary
        summary = self._summary(x_dict_pos)

        return x_dict_pos, x_dict_neg, summary
    
    def discriminate(
        self,
        x_dict: Dict[NodeType, torch.Tensor],
        summary: torch.Tensor,
        is_sigmoid: bool = True,
    ) -> Dict[str, torch.Tensor]:
        results = {}

        assert summary.shape[0] == self.channel

        for node_type, x in x_dict.items():
            if not x.shape[0]:
                # if no nodes in this type, skip it
                continue 
            assert x.shape[1] == self.channel
            w = self.mapper_dict[node_type].weight
            value = torch.matmul(x, torch.matmul(w, summary))
            results[node_type] = torch.sigmoid(value) if is_sigmoid else value
            
        return results

    def loss(
        self,
        x_dict_pos: Dict[NodeType, torch.Tensor],
        x_dict_neg: Dict[NodeType, torch.Tensor],
        summary: torch.Tensor,
    ):
        loss = []
        pos_score_dict = self.discriminate(x_dict_pos, summary)
        neg_score_dict = self.discriminate(x_dict_neg, summary)

        # aggregate
        for node_type in pos_score_dict.keys():
            pos_score = pos_score_dict[node_type]
            neg_score = neg_score_dict[node_type]
            pos_loss = -torch.log(pos_score + 1e-8).mean()
            neg_loss = -torch.log(1 - neg_score + 1e-8).mean()
            loss.append(pos_loss+neg_loss)
        
        return sum(loss) / len(loss)

    def _corruption(
        self,
        batch: HeteroData,
    ):
        """
        inplace to corrupt the batch
        """
        for _,  tf in batch.tf_dict.items():
            if self.corruption is not None:
                self.corruption(tf)
            else:
                self._shuffle_tensorframe(tf)

    def _shuffle_tensorframe(
        self,
        tf: TensorFrame
    ):
        """
        inplace to shuffle the tensorframe
        """
        shuffle_idx = torch.randperm(len(tf), device=tf.device)
        for key in tf.feat_dict.keys():
            tf.feat_dict[key] = tf.feat_dict[key][shuffle_idx]

    def _summary(
        self,
        x_dict: Dict[NodeType, torch.Tensor],
    ) -> torch.Tensor:
        # return [channel]
        summary_list = [x.mean(dim=0).sigmoid() for x in x_dict.values() if x.shape[0] > 0]
        summary = sum(summary_list)/ len(summary_list)
        # print(summary.shape)
        return summary
