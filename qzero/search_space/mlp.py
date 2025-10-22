from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import (
    BatchNorm1d,
    Dropout,
    LayerNorm,
    Linear,
    Module,
    ReLU,
    Sequential,
)

import torch_frame
from torch_frame import TensorFrame, stype
from torch_frame.data.stats import StatType
from torch_frame.nn.encoder.stype_encoder import (
    EmbeddingEncoder,
    LinearEncoder,
    StypeEncoder,
)
from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder


class QZeroMLP(Module):
    r"""Modified From  torch_frame.nn.models.mlp
        hidden_dims (list[int] | None): q_zero provided:Per-layer hidden sizes for the MLP.
            If provided, it must have length == num_layers - 1.
            If None, uses uniform `channels` per hidden layer (original behavior).
    """

    blocks_choices = [2, 3]
    channel_choices = [64, 128, 256]

    def __init__(
        self,
        channels: int,
        out_channels: int,
        num_layers: int,
        col_stats: dict[str, dict[StatType, Any]],
        col_names_dict: dict[torch_frame.stype, list[str]],
        stype_encoder_dict: dict[torch_frame.stype, StypeEncoder]
        | None = None,
        normalization: str | None = "layer_norm",
        dropout_prob: float = 0.2,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()

        if stype_encoder_dict is None:
            stype_encoder_dict = {
                stype.categorical: EmbeddingEncoder(),
                stype.numerical: LinearEncoder(),
            }

        self.encoder = StypeWiseFeatureEncoder(
            out_channels=channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
        )

        # ===== z_zero: customer hidden size =====
        if hidden_dims is not None:
            if len(hidden_dims) != max(num_layers - 1, 0):
                raise ValueError(
                    f"`hidden_dims` length ({len(hidden_dims)}) "
                    f"must equal `num_layers - 1` ({num_layers - 1})."
                )
            widths = hidden_dims
        else:
            widths = [channels] * max(num_layers - 1, 0)

        self.mlp = Sequential()
        in_dim = channels
        for out_dim in widths:
            self.mlp.append(Linear(in_dim, out_dim))
            if normalization == "layer_norm":
                self.mlp.append(LayerNorm(out_dim))
            elif normalization == "batch_norm":
                self.mlp.append(BatchNorm1d(out_dim))
            self.mlp.append(ReLU())
            self.mlp.append(Dropout(p=dropout_prob))
            in_dim = out_dim

        self.mlp.append(Linear(in_dim, out_channels))
        # ===== q_zero done =====

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()
        for param in self.mlp:
            if hasattr(param, 'reset_parameters'):
                param.reset_parameters()

    def forward(self, tf: TensorFrame) -> Tensor:
        r"""Transforming :class:`TensorFrame` object into output prediction.

        Args:
            tf (TensorFrame): Input :class:`TensorFrame` object.

        Returns:
            torch.Tensor: Output of shape [batch_size, out_channels].
        """
        x, _ = self.encoder(tf)

        x = torch.mean(x, dim=1)

        out = self.mlp(x)
        return out

    def forward_wo_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """q-zero
        x: [B, channels]，this is after encoder+mean-pool
        return: [B, out_channels]
        """
        return self.mlp(x)
