"""Regression tests for pointer probabilities, masking, and gradients."""

from __future__ import annotations

import unittest

import torch

from stage2.model import GraphPointerNetwork, make_node_features


class GraphPointerNetworkTests(unittest.TestCase):
    def test_pointer_probability_uses_encoded_dimension(self) -> None:
        """With two clients, only the first decision has nonzero log-P."""
        for hidden_dim in (8, 32, 128):
            with self.subTest(hidden_dim=hidden_dim):
                torch.manual_seed(1234)
                model = GraphPointerNetwork(hidden_dim, encoder_layers=1, heads=2)
                model.eval()
                features = torch.randn(2, 3, model.feature_count)

                with torch.no_grad():
                    encoded = model.encoder(model.embedding(features))
                    depot = encoded[:, 0]
                    query = model.query(depot, depot)
                    scores = (
                        model.key(encoded) * model.query_projection(query).unsqueeze(1)
                    ).sum(dim=-1) / hidden_dim**0.5
                    expected_log_probability = torch.log_softmax(scores[:, 1:], dim=-1).max(dim=-1).values
                    route, log_probability = model(features, decode_type="greedy")

                torch.testing.assert_close(log_probability, expected_log_probability, rtol=1e-5, atol=1e-6)
                self.assertEqual(route[:, 0].tolist(), (scores[:, 1:].argmax(dim=-1) + 1).tolist())

    def test_sampled_mixed_size_routes_allow_finite_backward(self) -> None:
        torch.manual_seed(1234)
        model = GraphPointerNetwork(hidden_dim=16, encoder_layers=1, heads=2)
        features = torch.randn(3, 5, model.feature_count)
        valid = torch.tensor([
            [True, True, True, True, True],
            [True, True, True, False, False],
            [True, True, False, False, False],
        ])

        route, log_probability = model(features, valid_node_mask=valid)
        (-log_probability.mean()).backward()

        self.assertTrue(torch.isfinite(log_probability).all().item())
        for row, client_count in enumerate((4, 2, 1)):
            self.assertEqual(sorted(route[row, :client_count].tolist()), list(range(1, client_count + 1)))
            self.assertEqual(route[row, client_count:].tolist(), [0] * (4 - client_count))
        gradient = model.embedding.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all().item())
        self.assertGreater(gradient.abs().sum().item(), 0.0)

    def test_greedy_mixed_size_routes_exclude_padding_and_depot(self) -> None:
        model = GraphPointerNetwork(hidden_dim=8, encoder_layers=1, heads=2)
        model.eval()
        features = torch.zeros(2, 4, model.feature_count)
        valid = torch.tensor([[True, True, True, True], [True, True, False, False]])

        with torch.no_grad():
            route, log_probability = model(features, decode_type="greedy", valid_node_mask=valid)

        self.assertEqual(sorted(route[0].tolist()), [1, 2, 3])
        self.assertEqual(route[1].tolist(), [1, 0, 0])
        self.assertAlmostEqual(log_probability[1].item(), 0.0)

    def test_existing_parameter_layout_loads_without_migration(self) -> None:
        original = GraphPointerNetwork(hidden_dim=8, encoder_layers=1, heads=2)
        restored = GraphPointerNetwork(hidden_dim=8, encoder_layers=1, heads=2)

        self.assertEqual(tuple(original.embedding.weight.shape), (8, 7))
        self.assertEqual(tuple(original.query.weight_ih.shape), (24, 8))
        self.assertEqual(tuple(original.key.weight.shape), (8, 8))

        result = restored.load_state_dict(original.state_dict(), strict=True)

        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])

    def test_invalid_features_and_masks_are_rejected(self) -> None:
        model = GraphPointerNetwork(hidden_dim=8, encoder_layers=1, heads=2)
        cases = [
            (torch.zeros(1, 3, 6), None, "features must have shape"),
            (torch.zeros(3, 7), None, "features must have shape"),
            (torch.zeros(1, 1, 7), None, "at least one client"),
            (torch.zeros(1, 3, 7), torch.ones(1, 2, dtype=torch.bool), "valid_node_mask must have shape"),
            (torch.zeros(1, 3, 7), torch.tensor([[False, True, True]]), "valid depot"),
            (torch.zeros(1, 3, 7), torch.tensor([[True, False, False]]), "at least one client"),
        ]
        for features, valid, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                model(features, valid_node_mask=valid)

    def test_node_features_remain_finite_with_zero_spatial_extent(self) -> None:
        coordinates = torch.zeros(1, 3, 2)

        features = make_node_features(coordinates, 6.0, 90.0, 12.0)

        self.assertEqual(tuple(features.shape), (1, 3, 7))
        self.assertTrue(torch.isfinite(features).all().item())
        self.assertEqual(features[0, :, 2].tolist(), [1.0, 0.0, 0.0])
        torch.testing.assert_close(features[0, :, 6], torch.full((3,), 12.0 / 23.0))


if __name__ == "__main__":
    unittest.main()
