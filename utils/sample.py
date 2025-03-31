
from relbench.base import EntityTask, Table, TaskType
from relbench.modeling.graph import NodeTrainTableInput, AttachTargetTransform
from relbench.modeling.utils import to_unix_time

import numpy as np
import torch
from torch import Tensor

from typing import Optional, Tuple, List


def get_node_train_table_input_with_sample(
    table: Table,
    task: EntityTask,
    sample_ratio: float = 1.0,
    shuffle: bool = True,
) -> Tuple[List[int], NodeTrainTableInput]:
    r"""Get the training table input for node prediction with sampled indices."""

    nodes = torch.from_numpy(table.df[task.entity_col].astype(int).values)

    # determine the number of samples to draw
    total_nodes = len(nodes)
    sample_size = max(1, min(int(total_nodes * sample_ratio), total_nodes))

    # sample indices without replacement
    if shuffle:
        sampled_indices = torch.randperm(total_nodes)[:sample_size].tolist()
    else:
        sampled_indices = list(range(sample_size))
        
    # select the sampled nodes
    sampled_nodes = nodes[sampled_indices]

    time: Optional[Tensor] = None
    if table.time_col is not None:
        time_values = torch.from_numpy(to_unix_time(table.df[table.time_col]))
        time = time_values[sampled_indices]

    target: Optional[Tensor] = None
    transform: Optional[AttachTargetTransform] = None
    if task.target_col in table.df:
        target_type = float
        if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            target_type = int
        if task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
            target_values = np.stack(table.df[task.target_col].values)
        else:
            target_values = table.df[task.target_col].values.astype(
                target_type)

        target = torch.from_numpy(target_values)[sampled_indices]
        transform = AttachTargetTransform(task.entity_table, target)

    return (sampled_indices, NodeTrainTableInput(
        nodes=(task.entity_table, sampled_nodes),
        time=time,
        target=target,
        transform=transform,
    ))
