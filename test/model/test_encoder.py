import unittest
import torch
from model.encoder import (
    build_encoder,
    TabMEncoder,
    MLPEncoder,
    ResNetEncoder,
    FTTransEncoder,
    DFMEncoder,
)


class TestBuildEncoder(unittest.TestCase):
    """Test suite for the unified build_encoder factory function."""

    def test_build_dfm_encoder(self):
        """Test building DFMEncoder."""
        encoder = build_encoder("dfm", channels=64, num_layers=2, dropout=0.2)
        self.assertIsInstance(encoder, DFMEncoder)
        self.assertEqual(encoder.channels, 64)

    def test_build_tabm_encoder(self):
        """Test building TabMEncoder."""
        encoder = build_encoder("tabm", channels=64, num_layers=2, k=8, dropout=0.2)
        self.assertIsInstance(encoder, TabMEncoder)
        self.assertEqual(encoder.channels, 64)
        self.assertEqual(encoder.k, 8)

    def test_build_mlp_encoder(self):
        """Test building MLPEncoder."""
        encoder = build_encoder("mlp", channels=64, num_layers=3, dropout=0.2)
        self.assertIsInstance(encoder, MLPEncoder)
        self.assertEqual(encoder.channels, 64)

    def test_build_resnet_encoder(self):
        """Test building ResNetEncoder."""
        encoder = build_encoder("resnet", channels=64, num_layers=4, dropout=0.2)
        self.assertIsInstance(encoder, ResNetEncoder)
        self.assertEqual(encoder.channels, 64)

    def test_build_transformer_encoder(self):
        """Test building FTTransEncoder."""
        encoder = build_encoder("transformer", channels=64, num_layers=2, num_heads=4, dropout=0.2)
        self.assertIsInstance(encoder, FTTransEncoder)
        self.assertEqual(encoder.channels, 64)
        self.assertEqual(encoder.num_heads, 4)

    def test_case_insensitive(self):
        """Test that encoder_type is case insensitive."""
        encoder1 = build_encoder("MLP", channels=32, num_layers=2)
        encoder2 = build_encoder("mlp", channels=32, num_layers=2)
        self.assertIsInstance(encoder1, MLPEncoder)
        self.assertIsInstance(encoder2, MLPEncoder)

    def test_invalid_encoder_type(self):
        """Test that invalid encoder_type raises ValueError."""
        with self.assertRaises(ValueError):
            build_encoder("invalid", channels=32, num_layers=2)

    def test_all_encoders_forward(self):
        """Test that all built encoders can forward."""
        x = torch.randn(4, 10, 32)
        encoder_types = ["dfm", "tabm", "mlp", "resnet", "transformer"]

        for enc_type in encoder_types:
            encoder = build_encoder(enc_type, channels=32, num_layers=2)
            encoder.eval()
            with torch.no_grad():
                output = encoder(x)
            self.assertEqual(output.shape, (4, 32),
                           f"build_encoder('{enc_type}') output shape mismatch")


class TestDFMEncoder(unittest.TestCase):
    """Test suite for DFMEncoder."""

    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        encoder = DFMEncoder(channels=64, num_layers=3, dropout_prob=0.2)
        encoder.eval()

        # Input: [batch_size, num_fields, channels]
        x = torch.randn(8, 10, 64)

        with torch.no_grad():
            output = encoder(x)

        # Output: [batch_size, channels]
        self.assertEqual(output.shape, (8, 64))

    def test_reset_parameters(self):
        """Test that reset_parameters method exists and works."""
        encoder = DFMEncoder(channels=32, num_layers=2, dropout_prob=0.1)
        encoder.reset_parameters()  # Should not raise error

    def test_backward(self):
        """Test backward pass."""
        encoder = DFMEncoder(channels=32, num_layers=2, dropout_prob=0.1)
        x = torch.randn(4, 5, 32, requires_grad=True)

        output = encoder(x)
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)


class TestTabMEncoder(unittest.TestCase):
    """Test suite for TabMEncoder."""

    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        encoder = TabMEncoder(channels=64, num_layers=3, k=8, dropout_prob=0.2)
        encoder.eval()

        # Input: [batch_size, num_columns, channels]
        x = torch.randn(8, 10, 64)

        with torch.no_grad():
            output = encoder(x)

        # Output: [batch_size, channels] after mean pooling
        self.assertEqual(output.shape, (8, 64))

    def test_reset_parameters(self):
        """Test that reset_parameters method exists and works."""
        encoder = TabMEncoder(channels=32, num_layers=2, k=4, dropout_prob=0.1)
        encoder.reset_parameters()  # Should not raise error

    def test_backward(self):
        """Test backward pass."""
        encoder = TabMEncoder(channels=32, num_layers=2, k=4, dropout_prob=0.1)
        x = torch.randn(4, 5, 32, requires_grad=True)

        output = encoder(x)
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)


class TestMLPEncoder(unittest.TestCase):
    """Test suite for MLPEncoder."""

    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        encoder = MLPEncoder(channels=64, num_layers=3, dropout=0.2)
        encoder.eval()

        # Input: [batch_size, num_nodes, channels]
        x = torch.randn(8, 10, 64)

        with torch.no_grad():
            output = encoder(x)

        # Output: [batch_size, channels] after mean pooling over nodes
        self.assertEqual(output.shape, (8, 64))

    def test_reset_parameters(self):
        """Test that reset_parameters method exists and works."""
        encoder = MLPEncoder(channels=32, num_layers=2, dropout=0.1)
        encoder.reset_parameters()  # Should not raise error

    def test_backward(self):
        """Test backward pass."""
        encoder = MLPEncoder(channels=32, num_layers=2, dropout=0.1)
        x = torch.randn(4, 5, 32, requires_grad=True)

        output = encoder(x)
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)


class TestResNetEncoder(unittest.TestCase):
    """Test suite for ResNetEncoder."""

    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        encoder = ResNetEncoder(channels=64, num_layers=4, dropout=0.2)
        encoder.eval()

        # Input: [batch_size, num_nodes, channels]
        x = torch.randn(8, 10, 64)

        with torch.no_grad():
            output = encoder(x)

        # Output: [batch_size, channels] after mean pooling over nodes
        self.assertEqual(output.shape, (8, 64))

    def test_reset_parameters(self):
        """Test that reset_parameters method exists and works."""
        encoder = ResNetEncoder(channels=32, num_layers=3, dropout=0.1)
        encoder.reset_parameters()  # Should not raise error

    def test_backward(self):
        """Test backward pass."""
        encoder = ResNetEncoder(channels=32, num_layers=2, dropout=0.1)
        x = torch.randn(4, 5, 32, requires_grad=True)

        output = encoder(x)
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)


class TestFTTransEncoder(unittest.TestCase):
    """Test suite for FTTransEncoder."""

    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        encoder = FTTransEncoder(channels=64, num_layers=2, num_heads=4, dropout=0.2)
        encoder.eval()

        # Input: [batch_size, seq_len, channels]
        x = torch.randn(8, 10, 64)

        with torch.no_grad():
            output = encoder(x)

        # Output: [batch_size, channels] after mean pooling over seq_len
        self.assertEqual(output.shape, (8, 64))

    def test_reset_parameters(self):
        """Test that reset_parameters method exists and works."""
        encoder = FTTransEncoder(channels=64, num_layers=2, num_heads=4, dropout=0.1)
        encoder.reset_parameters()  # Should not raise error

    def test_backward(self):
        """Test backward pass."""
        encoder = FTTransEncoder(channels=32, num_layers=1, num_heads=4, dropout=0.1)
        x = torch.randn(4, 5, 32, requires_grad=True)

        output = encoder(x)
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)

    def test_channels_divisible_by_heads(self):
        """Test assertion when channels not divisible by num_heads."""
        with self.assertRaises(AssertionError):
            FTTransEncoder(channels=65, num_layers=2, num_heads=4)


class TestEncoderAPI(unittest.TestCase):
    """Test that all encoders follow the same API."""

    def test_all_have_reset_parameters(self):
        """Test that all encoders have reset_parameters method."""
        encoders = [
            DFMEncoder(channels=64, num_layers=2, dropout_prob=0.2),
            TabMEncoder(channels=64, num_layers=2, k=4, dropout_prob=0.2),
            MLPEncoder(channels=64, num_layers=2, dropout=0.2),
            ResNetEncoder(channels=64, num_layers=2, dropout=0.2),
            FTTransEncoder(channels=64, num_layers=2, num_heads=4, dropout=0.2),
        ]

        for encoder in encoders:
            self.assertTrue(hasattr(encoder, 'reset_parameters'))
            encoder.reset_parameters()

    def test_all_accept_common_params(self):
        """Test that all encoders accept channels, num_layers."""
        # All should accept these parameters
        DFMEncoder(channels=32, num_layers=2, dropout_prob=0.1)
        TabMEncoder(channels=32, num_layers=2, k=4, dropout_prob=0.1)
        MLPEncoder(channels=32, num_layers=2, dropout=0.1)
        ResNetEncoder(channels=32, num_layers=2, dropout=0.1)
        FTTransEncoder(channels=32, num_layers=2, num_heads=4, dropout=0.1)

    def test_all_same_output_shape(self):
        """Test that all encoders output [batch, channels]."""
        x = torch.randn(4, 10, 32)

        encoders = [
            DFMEncoder(channels=32, num_layers=2, dropout_prob=0.1),
            TabMEncoder(channels=32, num_layers=2, k=4, dropout_prob=0.1),
            MLPEncoder(channels=32, num_layers=2, dropout=0.1),
            ResNetEncoder(channels=32, num_layers=2, dropout=0.1),
            FTTransEncoder(channels=32, num_layers=2, num_heads=4, dropout=0.1),
        ]

        for encoder in encoders:
            encoder.eval()
            with torch.no_grad():
                output = encoder(x)
            # All should output [batch, channels]
            self.assertEqual(output.shape, (4, 32),
                           f"{encoder.__class__.__name__} output shape mismatch")


if __name__ == '__main__':
    unittest.main()
