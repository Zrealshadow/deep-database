import unittest
import torch
import pandas as pd
from torch_frame.data import TensorFrame,Dataset
from torch_frame.data.stats import StatType
from torch_frame import stype
from model.tabular.tabm import TabMEnsemble, TabM


class TestTabMEnsemble(unittest.TestCase):
    """Test TabMEnsemble with a simple input example."""

    def test_forward_pass(self):
        """Test forward pass with example input."""
        # Model parameters
        batch_size = 16
        nfields = 10
        channels = 64
        num_layers = 3
        k = 8

        # Create model
        model = TabMEnsemble(
            nfields=nfields,
            channels=channels,
            num_layers=num_layers,
            k=k
        )
        model.reset_parameters()
        model.eval()

        # Create input tensor: (batch_size, k, nfields * channels)
        x = torch.randn(batch_size, nfields * channels)

        # Forward pass
        output = model(x)

        # Check output shape: (batch_size, channels)
        self.assertEqual(output.shape, (batch_size, channels))

        # Check no NaN or Inf
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())


class TestTabM(unittest.TestCase):
    """Test TabM full model with a simple input example."""

    def test_forward_pass(self):
        """Test forward pass with TensorFrame input."""
        # Model parameters
        batch_size = 16
        channels = 32
        out_channels = 5
        num_layers = 2
        k = 4

        # Create a simple DataFrame
        df = pd.DataFrame({
            'cat_0': ['A', 'B', 'C'] * (batch_size // 3) + ['A'] * (batch_size % 3),
            'cat_1': ['X', 'Y', 'Z'] * (batch_size // 3) + ['X'] * (batch_size % 3),
            'num_0': torch.randn(batch_size).tolist(),
            'num_1': torch.randn(batch_size).tolist(),
        })

        # Define column names and their types
        col_to_stype = {
            'cat_0': stype.categorical,
            'cat_1': stype.categorical,
            'num_0': stype.numerical,
            'num_1': stype.numerical,
        }

        # Create TensorFrame from DataFrame
        dataset = Dataset(df, col_to_stype).materialize()
        tf = dataset.tensor_frame
        # Create model
        model = TabM(
            channels=channels,
            out_channels=out_channels,
            num_layers=num_layers,
            col_stats= dataset.col_stats,
            col_names_dict= tf.col_names_dict,
            k=k
        )
        
        model.eval()

       
        # Forward pass
        output = model(tf)

        # Check output shape: (batch_size, out_channels)
        self.assertEqual(output.shape, (batch_size, out_channels))

        # Check no NaN or Inf
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())


if __name__ == '__main__':
    unittest.main()
