import unittest
import torch
from model.relationConv import AttentionHeteroConv, SharedMessagePassingConv


class TestSharedMessagePassingConv(unittest.TestCase):
    """Test SharedMessagePassingConv layer."""

    def test_forward_homogeneous(self):
        """Test forward pass on homogeneous graph."""
        in_channels = 16
        out_channels = 32
        num_nodes = 10
        num_aggr = 4  # MAX, MIN, SUM, MEAN
        shared_lin = torch.nn.Linear(in_channels, out_channels)
        conv = SharedMessagePassingConv(shared_lin)

        x = torch.randn(num_nodes, in_channels)
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 0]
        ])

        output = conv(x, edge_index)

        self.assertEqual(output.shape, (num_nodes, num_aggr, out_channels))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_forward_bipartite(self):
        """Test forward pass on bipartite graph."""
        in_channels = 16
        out_channels = 32
        num_src = 10
        num_dst = 5
        num_aggr = 4  # MAX, MIN, SUM, MEAN

        shared_lin = torch.nn.Linear(in_channels, out_channels)
        conv = SharedMessagePassingConv(shared_lin)

        x_src = torch.randn(num_src, in_channels)
        x_dst = torch.randn(num_dst, in_channels)
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4],
            [0, 1, 2, 0, 1]
        ])

        output = conv((x_src, x_dst), edge_index)

        self.assertEqual(output.shape, (num_dst, num_aggr, out_channels))
        self.assertFalse(torch.isnan(output).any())

    def test_backward_pass(self):
        """Test backward pass."""
        in_channels = 16
        out_channels = 32

        shared_lin = torch.nn.Linear(in_channels, out_channels)
        conv = SharedMessagePassingConv(shared_lin)

        x = torch.randn(10, in_channels, requires_grad=True)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])

        output = conv(x, edge_index)
        loss = output.sum()
        loss.backward()

        # Check gradients
        self.assertIsNotNone(shared_lin.weight.grad)
        self.assertIsNotNone(x.grad)

    def test_isolated_target_nodes(self):
        """Test that isolated target nodes get zero vectors from scatter."""
        in_channels = 16
        out_channels = 32
        num_aggr = 4  # MAX, MIN, SUM, MEAN

        shared_lin = torch.nn.Linear(in_channels, out_channels)
        conv = SharedMessagePassingConv(shared_lin)
        conv.eval()

        # 10 source nodes, 5 target nodes
        x_src = torch.randn(10, in_channels)
        x_dst = torch.randn(5, in_channels)

        # Only target nodes 0, 1, 2 receive edges
        # Target nodes 3, 4 are ISOLATED
        edge_index = torch.tensor([
            [0, 1, 2],  # source nodes
            [0, 1, 2]   # target nodes
        ])

        with torch.no_grad():
            output = conv((x_src, x_dst), edge_index)

        # Check output shape
        self.assertEqual(output.shape, (5, num_aggr, out_channels))

        # Nodes 0, 1, 2 should have non-zero outputs
        self.assertFalse(torch.allclose(output[0], torch.zeros_like(output[0])))
        self.assertFalse(torch.allclose(output[1], torch.zeros_like(output[1])))
        self.assertFalse(torch.allclose(output[2], torch.zeros_like(output[2])))

        # Isolated nodes 3, 4 should get ZERO vectors from scatter
        self.assertTrue(torch.allclose(output[3], torch.zeros_like(output[3])))
        self.assertTrue(torch.allclose(output[4], torch.zeros_like(output[4])))


class TestAttentionHeteroConv(unittest.TestCase):
    """Test AttentionHeteroConv layer."""

    def setUp(self):
        """Set up test fixtures."""
        self.edge_types = [
            ('user', 'buys', 'item'),
            ('user', 'views', 'item'),
            ('item', 'bought_by', 'user'),
        ]
        self.in_channels = 32
        self.out_channels = 64

    def test_initialization(self):
        """Test model initialization."""
        model = AttentionHeteroConv(
            edge_types=self.edge_types,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            num_heads=4
        )

        # Check that convs are created for each edge type
        self.assertEqual(len(model.convs), len(self.edge_types))

        # Check shared linear layers exist
        self.assertIsNotNone(model.lin_neighbor)
        self.assertIsNotNone(model.lin_self)
        self.assertIsNotNone(model.multihead_attn)

    def test_forward_pass(self):
        """Test forward pass with multiple edge types."""
        model = AttentionHeteroConv(
            edge_types=self.edge_types,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
        )
        model.eval()

        # Create input
        x_dict = {
            'user': torch.randn(10, self.in_channels),
            'item': torch.randn(5, self.in_channels)
        }

        edge_index_dict = {
            ('user', 'buys', 'item'): torch.tensor([
                [0, 1, 2, 3, 4],
                [0, 1, 2, 0, 1]
            ]),
            ('user', 'views', 'item'): torch.tensor([
                [0, 1, 5, 6],
                [0, 1, 2, 3]
            ]),
            ('item', 'bought_by', 'user'): torch.tensor([
                [0, 1, 2],
                [0, 1, 2]
            ]),
        }

        with torch.no_grad():
            out_dict, seq_types, attn_weights = model(x_dict, edge_index_dict)

        # Check output shapes
        self.assertEqual(out_dict['user'].shape, (10, self.out_channels))
        self.assertEqual(out_dict['item'].shape, (5, self.out_channels))

        # Check that seq_types and attn_weights are returned
        self.assertIsInstance(seq_types, dict)
        self.assertIsInstance(attn_weights, dict)

        # Check no NaN or Inf
        self.assertFalse(torch.isnan(out_dict['user']).any())
        self.assertFalse(torch.isnan(out_dict['item']).any())
        self.assertFalse(torch.isinf(out_dict['user']).any())
        self.assertFalse(torch.isinf(out_dict['item']).any())

    def test_single_edge_type(self):
        """Test with only one edge type (no attention)."""
        edge_types = [('user', 'follows', 'user')]

        model = AttentionHeteroConv(
            edge_types=edge_types,
            in_channels=16,
            out_channels=32,
        )
        model.eval()

        x_dict = {'user': torch.randn(8, 16)}
        edge_index_dict = {
            ('user', 'follows', 'user'): torch.tensor([
                [0, 1, 2, 3],
                [1, 2, 3, 0]
            ])
        }

        with torch.no_grad():
            out_dict, _, _ = model(x_dict, edge_index_dict)

        self.assertEqual(out_dict['user'].shape, (8, 32))

    def test_isolated_nodes(self):
        """Test nodes with no incoming edges."""
        edge_types = [('user', 'likes', 'item')]

        model = AttentionHeteroConv(
            edge_types=edge_types,
            in_channels=16,
            out_channels=32,
        )
        model.eval()

        # 10 users, 5 items, but only 2 items receive edges
        x_dict = {
            'user': torch.randn(10, 16),
            'item': torch.randn(5, 16)
        }

        edge_index_dict = {
            ('user', 'likes', 'item'): torch.tensor([
                [0, 1, 2],
                [0, 1, 0]  # only items 0 and 1 receive edges
            ])
        }

        with torch.no_grad():
            out_dict, _, _ = model(x_dict, edge_index_dict)

        # All items should have output (including isolated ones)
        self.assertEqual(out_dict['item'].shape, (5, 32))

        # Users have no incoming edges
        self.assertEqual(out_dict['user'].shape, (10, 32))

    def test_backward_pass(self):
        """Test backward pass and gradient computation."""
        model = AttentionHeteroConv(
            edge_types=self.edge_types,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
        )

        x_dict = {
            'user': torch.randn(10, self.in_channels),
            'item': torch.randn(5, self.in_channels)
        }

        edge_index_dict = {
            ('user', 'buys', 'item'): torch.tensor([[0, 1], [0, 1]]),
            ('user', 'views', 'item'): torch.tensor([[0, 1], [0, 1]]),
            ('item', 'bought_by', 'user'): torch.tensor([[0, 1], [0, 1]]),
        }

        out_dict, _, _ = model(x_dict, edge_index_dict)
        loss = out_dict['user'].sum() + out_dict['item'].sum()
        loss.backward()

        # Check that parameters have gradients
        self.assertIsNotNone(model.lin_neighbor.weight.grad)
        self.assertIsNotNone(model.lin_self.weight.grad)

        # MultiheadAttention has multiple weight parameters
        # Check in_proj_weight (combined Q, K, V projection)
        self.assertIsNotNone(model.multihead_attn.in_proj_weight.grad)
        self.assertIsNotNone(model.multihead_attn.out_proj.weight.grad)

    def test_attention_aggregation(self):
        """Test that attention is applied for multiple edge types."""
        edge_types = [
            ('user', 'buys', 'item'),
            ('user', 'views', 'item'),
        ]

        model = AttentionHeteroConv(
            edge_types=edge_types,
            in_channels=16,
            out_channels=32,
        )
        model.eval()

        x_dict = {
            'user': torch.randn(5, 16),
            'item': torch.randn(3, 16)
        }

        # Items receive from both edge types
        edge_index_dict = {
            ('user', 'buys', 'item'): torch.tensor([[0, 1], [0, 1]]),
            ('user', 'views', 'item'): torch.tensor([[2, 3], [0, 1]]),
        }

        with torch.no_grad():
            out_dict, _, _ = model(x_dict, edge_index_dict)

        # Items 0 and 1 receive from both edge types (attention applied)
        # Item 2 receives no edges (only self features)
        self.assertEqual(out_dict['item'].shape, (3, 32))

    def test_isolated_nodes_get_self_features(self):
        """Test attention weights and sequence types for isolated vs connected nodes."""
        edge_types = [('user', 'likes', 'item')]

        model = AttentionHeteroConv(
            edge_types=edge_types,
            in_channels=16,
            out_channels=32,
        )
        model.eval()

        x_dict = {
            'user': torch.randn(5, 16),
            'item': torch.randn(5, 16)
        }

        # Only items 0, 1 receive edges
        # Items 2, 3, 4 are ISOLATED
        edge_index_dict = {
            ('user', 'likes', 'item'): torch.tensor([
                [0, 1, 2],
                [0, 1, 0]  # only items 0 and 1
            ])
        }

        with torch.no_grad():
            out_dict, seq_types, attn_weights = model(x_dict, edge_index_dict, require_attn_weights=True)

        # Check sequence types exist
        self.assertIn('item', seq_types)
        self.assertIn('user', seq_types)

        # Check attention weights exist and are not None
        self.assertIn('item', attn_weights)
        self.assertIn('user', attn_weights)
        self.assertIsNotNone(attn_weights['item'])
        self.assertIsNotNone(attn_weights['user'])

        # Check output shapes
        self.assertEqual(out_dict['item'].shape, (5, 32))
        self.assertEqual(out_dict['user'].shape, (5, 32))

        # Check that attention weights have correct shape
        # For items, all nodes get attention (even isolated ones process self token)
        # attn_weights shape: [num_nodes, seq_len] where seq_len varies per node
        item_attn = attn_weights['item']
        self.assertEqual(item_attn.shape[0], 5)  # 5 item nodes

        # Check no NaN or Inf in outputs
        self.assertFalse(torch.isnan(out_dict['item']).any())
        self.assertFalse(torch.isinf(out_dict['item']).any())
        self.assertFalse(torch.isnan(out_dict['user']).any())
        self.assertFalse(torch.isinf(out_dict['user']).any())
        


if __name__ == '__main__':
    unittest.main()
