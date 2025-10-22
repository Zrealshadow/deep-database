"""
Search space models for zero-cost NAS

Provides custom MLP and ResNet models with flexible architectures.
"""

from .mlp import QZeroMLP
from .resnet import QZeroResNet

__all__ = ["QZeroMLP", "QZeroResNet"]

