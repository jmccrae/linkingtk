"""Unit tests for [linkingtk.algorithms.ea._ea_losses][]."""

from __future__ import annotations

import torch

from linkingtk.algorithms.ea._ea_losses import (
    margin_ranking_loss_l1,
    margin_ranking_loss_l2_squared,
)


class TestMarginRankingLossL1:
    def test_zero_when_positives_coincide_and_negatives_are_far(self) -> None:
        embeddings = torch.tensor(
            [[0.0, 0.0], [0.0, 0.0], [10.0, 10.0], [10.0, 10.0]], dtype=torch.float32
        )
        pos_left = torch.tensor([0])
        pos_right = torch.tensor([1])
        neg_left = torch.tensor([0])
        neg_right = torch.tensor([2])
        neg2_left = torch.tensor([3])
        neg2_right = torch.tensor([1])

        loss = margin_ranking_loss_l1(
            embeddings, pos_left, pos_right, neg_left, neg_right, neg2_left, neg2_right, gamma=1.0
        )

        assert loss.item() == 0.0

    def test_positive_when_negatives_are_close(self) -> None:
        embeddings = torch.tensor([[0.0], [0.1], [0.1], [0.1]], dtype=torch.float32)
        pos_left = torch.tensor([0])
        pos_right = torch.tensor([1])
        neg_left = torch.tensor([0])
        neg_right = torch.tensor([2])
        neg2_left = torch.tensor([3])
        neg2_right = torch.tensor([1])

        loss = margin_ranking_loss_l1(
            embeddings, pos_left, pos_right, neg_left, neg_right, neg2_left, neg2_right, gamma=1.0
        )

        assert loss.item() > 0.0

    def test_gradient_flows_to_embeddings(self) -> None:
        embeddings = torch.randn(4, 3, requires_grad=True)
        pos_left = torch.tensor([0])
        pos_right = torch.tensor([1])
        neg_left = torch.tensor([0, 0])
        neg_right = torch.tensor([2, 3])
        neg2_left = torch.tensor([2, 3])
        neg2_right = torch.tensor([1, 1])

        loss = margin_ranking_loss_l1(
            embeddings, pos_left, pos_right, neg_left, neg_right, neg2_left, neg2_right, gamma=1.0
        )
        loss.backward()

        assert embeddings.grad is not None


class TestMarginRankingLossL2Squared:
    def test_zero_when_positives_coincide_and_negatives_are_far(self) -> None:
        embeddings = torch.tensor(
            [[0.0, 0.0], [0.0, 0.0], [10.0, 10.0], [10.0, 10.0]], dtype=torch.float32
        )
        pos_left = torch.tensor([0])
        pos_right = torch.tensor([1])
        neg_left = torch.tensor([0])
        neg_right = torch.tensor([2])

        loss = margin_ranking_loss_l2_squared(
            embeddings, pos_left, pos_right, neg_left, neg_right, margin=1.0, neg_margin_balance=0.1
        )

        assert loss.item() == 0.0

    def test_positive_when_negatives_are_close(self) -> None:
        embeddings = torch.tensor([[0.0], [0.5], [0.5], [0.5]], dtype=torch.float32)
        pos_left = torch.tensor([0])
        pos_right = torch.tensor([1])
        neg_left = torch.tensor([2])
        neg_right = torch.tensor([3])

        loss = margin_ranking_loss_l2_squared(
            embeddings, pos_left, pos_right, neg_left, neg_right, margin=1.0, neg_margin_balance=0.1
        )

        assert loss.item() > 0.0
