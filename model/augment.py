
import torch
import random

from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType
from torch_frame.data import TensorFrame

from typing import Dict, Callable, Optional, List

from enum import Enum


class Pemutation(Enum):
    DROP_NODE = 1
    DROP_EDGE = 2
    MASK_ATTRIBUTE = 3
    SUBGRAPH = 4

PERMUTATIONS = [
    Pemutation.DROP_NODE,
    Pemutation.DROP_EDGE,
    Pemutation.MASK_ATTRIBUTE,
    Pemutation.SUBGRAPH,
]


class HeteroGraphPermutationTool(object):

    @staticmethod
    def random_permutation(
        batch: HeteroData,
        drop_prob: float = 0.1,
        exclude_edge_types: List[str] = [],
        exclude_node_types: List[str] = [],
        seed: Optional[int] = None,
        inplace: bool = False,
        verbose: bool = False,
    ) -> HeteroData:

        """
        Randomly permute the nodes and edges in the graph.

        Args:
            batch (HeteroData): The input graph data.
            drop_prob (float, optional): Probability of dropping an edge. Defaults to 0.1.
            exclude_edge_types (List[str], optional): List of edge types to exclude from permutation. Defaults to [].
            exclude_node_types (List[str], optional): List of node types to exclude from permutation. Defaults to [].
            seed (Optional[int], optional): Random seed for reproducibility. Defaults to None.
            inplace (bool, optional): If True, modify the input data in place. Defaults to False.
            verbose (bool, optional): If True, print additional information. Defaults to False.

        Returns:
            HeteroData: The modified graph data with permuted nodes and edges.
        """

        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
        
        ptype = random.choice(PERMUTATIONS)

        if ptype == Pemutation.DROP_NODE:
            return HeteroGraphPermutationTool.random_node_drop(
                batch,
                drop_prob=drop_prob,
                exclude_node_types=exclude_node_types,
                seed=seed,
                inplace=inplace,
                verbose=verbose,
            )
        elif ptype == Pemutation.DROP_EDGE:
            return HeteroGraphPermutationTool.random_edge_drop(
                batch,
                drop_prob=drop_prob,
                exclude_edge_types=exclude_edge_types,
                seed=seed,
                inplace=inplace,
                verbose=verbose,
            )
        elif ptype == Pemutation.MASK_ATTRIBUTE:
            pass
        elif ptype == Pemutation.SUBGRAPH:
            pass
        return batch
    
    @staticmethod
    def feature_shuffle(
        self,
        batch: HeteroData,
        inplace: bool = False,
    ) -> HeteroData:
        """
        Shuffle the features of the nodes in the graph.

        Args:
            batch (HeteroData): The input graph data.
            inplace (bool, optional): If True, modify the input data in place. Defaults to False.

        Returns:
            HeteroData: The modified graph data with shuffled features.
        """

        if not inplace:
            batch = batch.clone()

        for _, tf in batch.tf_dict.items():
            shuffle_idx = torch.randperm(len(tf), device=tf.device)
            for key in tf.feat_dict.keys():
                tf.feat_dict[key] = tf.feat_dict[key][shuffle_idx]

        return batch

    @staticmethod
    def random_edge_drop(
        data: HeteroData,
        drop_prob: float = 0.1,
        exclude_edge_types: List[str] = [],
        seed: Optional[int] = None,
        inplace: bool = False,
        verbose: bool = False,
    ) -> HeteroData:
        """ Randomly remove edges from the graph.
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
        assert data.validate()
        return data

    @staticmethod
    def random_node_drop(
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
        assert data.validate()
        return data
