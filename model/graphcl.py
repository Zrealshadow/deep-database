

import torch
import random

from model.base import CompositeModel
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType
from torch_frame.data import TensorFrame

from typing import Dict, Callable, Optional, List


class HeteroGraphCL(torch.nn.Module):
    """reference to the implementation of GraphCL in pyg-ssl
    https://github.com/iDEA-iSAIL-Lab-UIUC/pyg-ssl/blob/main/src/methods/graphcl.py

    it is based on DGI, but the graph hidden vector is from enhanced graph.
    NOTE: it is different from the original description in GraphCL paper.
    """

    def __init__(
        self,
        data: HeteroData,
        channel: int,
        encoder: CompositeModel,
    ):

        super().__init__()
        self.channel = channel
        self.encoder = encoder
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
    
    def augment_heterodata(
        self,
        batch: HeteroData,
        drop_prob: float = 0.1,
        exclude_edge_types: List[str] = [],
        exclude_node_types: List[str] = [],
        seed: Optional[int] = None,
        inplace: bool = False,
        verbose: bool = False,
    ):
        
        if random.random() < 0.5:
            # drop edges
            return self._permutation_drop_edge(
                batch,
                drop_prob=drop_prob,
                exclude_edge_types=exclude_edge_types,
                seed=seed,
                inplace=inplace,
                verbose=verbose,
            )
        else:
            # drop nodes
            return self._permutation_drop_node(
                batch,
                drop_prob=drop_prob,
                exclude_node_types=exclude_node_types,
                seed=seed,
                inplace=inplace,
                verbose=verbose,
            )
    
    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType,
    ):
        aug_batch = self.augment_heterodata(
            batch,
            drop_prob=0.1,
            exclude_node_types=[entity_table],
            inplace=False,
        )
        # random permutation
        # corruption
        x_dict_pos = self.encoder.get_node_embedding(batch, entity_table)
        x_dict_aug = self.encoder.get_node_embedding(aug_batch, entity_table)

        self._corrpution(batch, inplace=True)
        x_dict_neg = self.encoder.get_node_embedding(batch, entity_table)

        summary = self._summary(x_dict_aug)

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

    def _corrpution(
        self,
        batch: HeteroData,
        inplace: bool = False,
    ) -> HeteroData:
        """Corruption the subgraph to generate a negative sample.
        # shuffle the node features.
        """

        if not inplace:
            batch = batch.clone()

        for _, tf in batch.tf_dict.items():
            shuffle_idx = torch.randperm(len(tf), device=tf.device)
            for key in tf.feat_dict.keys():
                tf.feat_dict[key] = tf.feat_dict[key][shuffle_idx]

        return batch

    def _summary(
        self,
        x_dict: Dict[NodeType, torch.Tensor],
    ) -> torch.Tensor:
        """return the summary of mini-batch graph
        average pooling all nodes in subgraph
        """
        summary_list = [x.mean(dim=0).sigmoid()
                        for x in x_dict.values() if x.shape[0] > 0]
        summary = sum(summary_list) / len(summary_list)
        # print(summary.shape)
        return summary

    def _permutation_drop_edge(
        self,
        data: HeteroData,
        drop_prob: float = 0.1,
        exclude_edge_types: List[str] = [],
        seed: Optional[int] = None,
        inplace: bool = False,
        verbose: bool = False,
    ) -> HeteroData:
        """ Randomly drop edges from the graph with a given probability.
        data -> batch from NeighborLoader
        """
        if seed is not None:
            torch.manual_seed(seed)

        if not inplace:
            data = data.clone()

        for edge_type in data.edge_types:
            if edge_type in exclude_edge_types:
                continue

            num_edges = data[edge_type].edge_index.shape[1]
            device = data[edge_type].edge_index.device
            keep_mask = torch.rand(num_edges, device=device) > drop_prob

            # apply the mask to the edge index
            data[edge_type].edge_index = data[edge_type].edge_index[:, keep_mask]

            # update 'e_id' and 'num_sampled_edges‘
            data[edge_type].e_id = data[edge_type].e_id[keep_mask]
            # NOTE: this step is no correct.
            # Not sure the num_sampled_edges is not used in sebsequent steps.
            # So we just set it to the number of edges after dropping
            data[edge_type].num_sampled_edges = keep_mask.sum()

            if verbose:
                print(
                    f"Drop {edge_type}: {num_edges} -> {data[edge_type].edge_index.shape[1]} edges"
                )

        # validate
        data.validate()
        return data

    def _permutation_drop_node(
        self,
        data: HeteroData,
        drop_prob: float = 0.1,
        exclude_node_types: List[str] = [],
        seed: Optional[int] = None,
        inplace: bool = False,
        verbose: bool = False,
    ) -> HeteroData:
        """ Randomly drop nodes from the graph with a given probability.
        """
        if seed is not None:
            torch.manual_seed(seed)

        if not inplace:
            data = data.clone()

        node_type_to_idx_map = {}
        exclude_node_types = set(exclude_node_types)
        for node_type in data.node_types:
            if node_type in exclude_node_types:
                continue
            tf: TensorFrame = data[node_type].tf
            num_nodes = tf.num_rows
            keep_mask = torch.rand(num_nodes, device=tf.device) > drop_prob
            # map old indices to new
            idx_map = torch.full(
                (num_nodes,), -1, dtype=torch.long, device=tf.device)
            idx_map[keep_mask] = torch.arange(
                keep_mask.sum(), device=tf.device)
            node_type_to_idx_map[node_type] = idx_map

            # update the batch
            # update other attributes
            for key, value in data[node_type].to_dict().items():
                if key in ["time", "n_id", "tf", "batch"]:
                    data[node_type][key] = value[keep_mask]
                elif key == "n_sampled_nodes":
                    data[node_type][key] = keep_mask.sum()
                # n_sampled_nodes are not used in the model
                # the node_type with input_id attribute is excluded when drop nodes

            # NOTE: num_sampled_nodes is corrupt
            data[node_type].num_sampled_nodes = keep_mask.sum()

            if verbose:
                print(
                    f"Drop {node_type}: {num_nodes} -> {keep_mask.sum()} nodes"
                )

        # update edges
        for edge_type in data.edge_types:
            src, _, dst = edge_type
            edge_index = data[edge_type].edge_index
            num_edge = edge_index.shape[1]
            src_idx = edge_index[0]
            dst_idx = edge_index[1]

            # update
            src_idx = node_type_to_idx_map[src][src_idx] if src in node_type_to_idx_map else src_idx
            dst_idx = node_type_to_idx_map[dst][dst_idx] if dst in node_type_to_idx_map else dst_idx

            mask = (src_idx > -1) & (dst_idx > -1)
            src_idx, dst_idx = src_idx[mask], dst_idx[mask]
            data[edge_type].edge_index = torch.stack(
                [src_idx, dst_idx], dim=0
            )

            # update 'e_id' and 'num_sampled_edges‘
            data[edge_type].e_id = data[edge_type].e_id[mask]

            # NOTE: this step is no correct.
            # Not sure the num_sampled_edges is not used in sebsequent steps.
            # So we just set it to the number of edges after dropping
            data[edge_type].num_sampled_edges = mask.sum()

            if verbose:
                print(
                    f"Drop {edge_type}: {num_edge} -> {mask.sum()} edges"
                )

        # validate
        data.validate()
        return data

    ''' ----------------------- NOT USED --------------------- '''

    def _readout_(
        self,
        batch: HeteroData,
        subgraph_num: int,
        x_dict: Dict[NodeType, torch.Tensor],
    ) -> torch.Tensor:
        """
        return the summary of mini-batch graph
        [B, C]
        B is the batch_size
        C is the hidden dimension
        # average pooling all nodes in subgraph
        NOTE: used for original GraphCL, not in this version
        """
        groups = [[] for _ in range(subgraph_num)]
        for node_type in batch.node_types:
            for i, bin_id in enumerate(batch[node_type].batch):
                groups[bin_id.item()].append(x_dict[node_type][i])
        groups = torch.stack([torch.stack(group).mean(dim=0)
                             for group in groups])
        return groups  # [B, C]
