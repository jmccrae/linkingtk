from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from linkingtk.algorithms.wsd._ewiser_structured_logits import StructuredLogits  # noqa: E402


def _adjacency(rows: list[int], cols: list[int], values: list[float], size: int) -> torch.Tensor:
    indices = torch.tensor([rows, cols], dtype=torch.long)
    return torch.sparse_coo_tensor(indices, torch.tensor(values), size=(size, size))


class TestForward:
    def test_propagates_neighbor_logits_with_residual(self) -> None:
        # node 0 depends fully on node 1 (weight 1.0); node 2 has no incoming edges.
        adjacency = _adjacency(rows=[0], cols=[1], values=[1.0], size=3)
        module = StructuredLogits(adjacency)
        logits = torch.tensor([[1.0, 10.0, 100.0]])

        out = module(logits)

        assert out[0, 0].item() == pytest.approx(1.0 + 10.0)  # residual + neighbor(node 1)
        assert out[0, 1].item() == pytest.approx(10.0)  # no incoming edges -> unchanged
        assert out[0, 2].item() == pytest.approx(100.0)  # no incoming edges -> unchanged

    def test_preserves_leading_batch_dimensions(self) -> None:
        adjacency = _adjacency(rows=[0], cols=[1], values=[1.0], size=3)
        module = StructuredLogits(adjacency)
        logits = torch.randn(2, 5, 3)

        out = module(logits)

        assert out.shape == logits.shape

    def test_renormalize_divides_by_row_sum(self) -> None:
        # node 0 has two incoming edges, weights 1.0 and 3.0 (row sum 4.0).
        adjacency = _adjacency(rows=[0, 0], cols=[1, 2], values=[1.0, 3.0], size=3)
        module = StructuredLogits(adjacency, renormalize=True)
        logits = torch.tensor([[0.0, 10.0, 10.0]])

        out = module(logits)

        # neighbor sum = 1*10 + 3*10 = 40, divided by row sum 4.0 -> 10.0, + residual 0.0
        assert out[0, 0].item() == pytest.approx(10.0)


class TestTrainable:
    def test_trainable_values_get_gradients(self) -> None:
        adjacency = _adjacency(rows=[0], cols=[1], values=[1.0], size=2)
        module = StructuredLogits(adjacency, trainable=True)
        logits = torch.tensor([[1.0, 2.0]])

        module(logits).sum().backward()

        assert module.adjacency_pars[1].grad is not None

    def test_non_trainable_values_get_no_gradients(self) -> None:
        adjacency = _adjacency(rows=[0], cols=[1], values=[1.0], size=2)
        module = StructuredLogits(adjacency, trainable=False)

        assert module.adjacency_pars[1].requires_grad is False


class TestStateDict:
    def test_keys_match_checkpoint_convention(self) -> None:
        adjacency = _adjacency(rows=[0], cols=[1], values=[1.0], size=2)
        module = StructuredLogits(adjacency)

        assert set(module.state_dict().keys()) == {
            "adjacency_pars.0",
            "adjacency_pars.1",
            "adjacency_pars.2",
        }

    def test_loads_from_a_plain_state_dict(self) -> None:
        # Mirrors real checkpoint loading: same edge structure (indices/size),
        # different values -- load_state_dict just needs matching shapes.
        source = StructuredLogits(_adjacency(rows=[0], cols=[1], values=[0.75], size=2))
        target = StructuredLogits(_adjacency(rows=[0], cols=[1], values=[0.0], size=2))

        target.load_state_dict(source.state_dict())

        assert torch.equal(target.adjacency_pars[1], source.adjacency_pars[1])
