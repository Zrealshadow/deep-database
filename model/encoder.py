import torch
from typing import Literal

from .tabular.layers import FactorizationMachine
from .tabular.tabm import BatchEnsembleLayer


def build_encoder(
    encoder_type: Literal["dfm", "tabm", "mlp", "resnet", "fttrans"],
    channels: int,
    num_layers: int,
    dropout: float = 0.2,
    **kwargs
):
    """
    Unified factory function to build encoders.

    Args:
        encoder_type: Type of encoder ("dfm", "tabm", "mlp", "resnet", "transformer")
        channels: Input and output feature dimension
        num_layers: Number of layers
        dropout: Dropout probability
        **kwargs: Additional encoder-specific arguments:
            - k (int): Ensemble size for TabMEncoder (default: 8)
            - activation (str): Activation function for TabMEncoder (default: "relu")
            - num_heads (int): Number of attention heads for FTTransEncoder (default: 4)

    Returns:
        An encoder module

    Examples:
        >>> encoder = build_encoder("mlp", channels=64, num_layers=3, dropout=0.2)
        >>> encoder = build_encoder("tabm", channels=64, num_layers=2, k=8)
        >>> encoder = build_encoder("transformer", channels=64, num_layers=2, num_heads=4)
    """
    encoder_type = encoder_type.lower()

    if encoder_type == "dfm":
        return DFMEncoder(
            channels=channels,
            num_layers=num_layers,
            dropout_prob=dropout,
        )
    elif encoder_type == "tabm":
        k = kwargs.get("k", 8)
        activation = kwargs.get("activation", "relu")
        return TabMEncoder(
            channels=channels,
            num_layers=num_layers,
            k=k,
            dropout_prob=dropout,
            activation=activation,
        )
    elif encoder_type == "mlp":
        return MLPEncoder(
            channels=channels,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif encoder_type == "resnet":
        return ResNetEncoder(
            channels=channels,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif encoder_type == "fttrans":
        num_heads = kwargs.get("num_heads", 4)
        return FTTransEncoder(
            channels=channels,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
    else:
        raise ValueError(
            f"Unknown encoder_type: {encoder_type}. "
            f"Supported types: 'dfm', 'tabm', 'mlp', 'resnet', 'transformer'"
        )


class TabMEncoder(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        num_layers: int,
        k: int = 8,
        dropout_prob: float = 0.2,
        activation: str = "relu",
    ):
        super().__init__()
        self.channels = channels
        self.layers = torch.nn.ModuleList()
        self.k = k
        for _ in range(num_layers):
            layer = BatchEnsembleLayer(
                in_channels=channels,
                out_channels=channels,
                k=k,
                dropout_prob=dropout_prob,
                activation=activation,
            )
            self.layers.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_columns, channels]

        Returns:
            x : [batch_size, channels]
        """
        x = x.mean(dim=1)
        # expand x to (batch_size, k, channels)
        x = x.unsqueeze(1).expand(-1, self.k, -1)
        for layer in self.layers:
            x = layer(x)  # [batch_size, k, channels]
        x = x.mean(dim=1)  # [batch_size, channels]
        return x

    def reset_parameters(self) -> None:
        for layer in self.layers:
            layer.reset_parameters()


class MLPEncoder(torch.nn.Module):
    def __init__(self, channels: int, num_layers: int, dropout: float = 0.2):
        """
        MLP Encoder for node features with same input/output dimensions.

        Args:
            channels: Input and output feature dimension
            num_layers: Number of layers in the MLP
            dropout: Dropout probability
        """
        super().__init__()
        self.channels = channels
        self.num_layers = num_layers

        layers = []
        for _ in range(num_layers):
            layers.append(torch.nn.Linear(channels, channels))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))

        self.mlp = torch.nn.Sequential(*layers)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_nodes, channels]

        Returns:
            x : [batch_size, channels]
        """
        x = x.mean(dim=1)
        return self.mlp(x)


class ResNetEncoder(torch.nn.Module):
    def __init__(self, channels: int, num_layers: int, dropout: float = 0.2):
        """
        ResNet-style Encoder for node features with same input/output dimensions.

        Args:
            channels: Input and output feature dimension
            num_layers: Number of residual blocks
            dropout: Dropout probability
        """
        super().__init__()
        self.channels = channels
        self.num_layers = num_layers

        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(channels, channels),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(p=dropout),
                    torch.nn.Linear(channels, channels),
                )
            )
        self.dropout = torch.nn.Dropout(p=dropout)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            for sublayer in layer:
                if isinstance(sublayer, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(sublayer.weight)
                    if sublayer.bias is not None:
                        torch.nn.init.zeros_(sublayer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_nodes, channels]

        Returns:
            x : [batch_size, channels]
        """
        x = x.mean(dim=1)
        for layer in self.layers:
            residual = x
            x = layer(x)
            x = x + residual  # Residual connection
            x = self.dropout(x)
        return x


class FTTransEncoder(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        num_layers: int,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        """
        Transformer-based Encoder for node features with same input/output dimensions.

        Args:
            channels: Input and output feature dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.channels = channels
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Ensure channels is divisible by num_heads
        assert channels % num_heads == 0, f"channels ({channels}) must be divisible by num_heads ({num_heads})"

        # Transformer encoder layers
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=channels * 4,  # Standard transformer feedforward dimension
            dropout=dropout,
            activation='relu',
            batch_first=True,
        )
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        self.reset_parameters()

    def reset_parameters(self):
        # TransformerEncoder has its own initialization
        for module in self.transformer.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, channels]

        Returns:
            x: [batch_size, channels]
        """
        x = self.transformer(x)
        # Mean pool over sequence dimension to get [batch_size, channels]
        x = x.mean(dim=1)
        return x

class DFMEncoder(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        num_layers: int,
        dropout_prob: float = 0.2,
    ):
        super().__init__()
        self.channels = channels
        self.linear = torch.nn.Linear(channels, channels)
        self.mlp = MLPEncoder(
            channels=channels,
            num_layers=num_layers,
            dropout=dropout_prob,
        )
        self.fm = FactorizationMachine(reduce_dim=False, normalize=True)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)
        self.mlp.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_fields, channels]

        Returns:
            x : [batch_size, channels]
        """
        linear_out = self.linear(x.mean(dim=1))  # [batch_size, channels]
        mlp_out = self.mlp(x)                    # [batch_size, channels]
        fm_out = self.fm(x)                      # [batch_size, channels]

        out = linear_out + mlp_out + fm_out
        return out
