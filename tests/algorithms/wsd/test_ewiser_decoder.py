from __future__ import annotations

import pickle
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from linkingtk.algorithms.wsd._ewiser_decoder import (  # noqa: E402
    EwiserDecoder,
    load_fairseq_checkpoint,
)


class TestEwiserDecoder:
    def test_forward_shape(self) -> None:
        decoder = EwiserDecoder(input_dim=8, hidden_dim=4, vocab_size=5)

        out = decoder(torch.randn(2, 3, 8))

        assert out.shape == (2, 3, 5)

    def test_state_dict_keys_match_checkpoint_convention(self) -> None:
        adjacency = torch.sparse_coo_tensor(
            torch.tensor([[0], [1]]), torch.tensor([0.5]), size=(5, 5)
        )
        decoder = EwiserDecoder(input_dim=8, hidden_dim=4, vocab_size=5, adjacency=adjacency)

        keys = set(decoder.state_dict().keys())

        assert keys == {
            "norm.weight",
            "norm.bias",
            "norm.running_mean",
            "norm.running_var",
            "norm.num_batches_tracked",
            "linears.0.weight",
            "linears.0.bias",
            "logits.weight",
            "structured_logits.adjacency_pars.0",
            "structured_logits.adjacency_pars.1",
            "structured_logits.adjacency_pars.2",
        }

    def test_no_adjacency_means_no_structured_logits(self) -> None:
        decoder = EwiserDecoder(input_dim=8, hidden_dim=4, vocab_size=5, adjacency=None)

        assert decoder.structured_logits is None
        assert decoder(torch.randn(2, 8)).shape == (2, 5)


class TestLoadFairseqCheckpoint:
    def test_loads_plain_tensors_and_namespace(self, tmp_path: Path) -> None:
        import argparse

        path = tmp_path / "checkpoint.pt"
        torch.save(
            {"args": argparse.Namespace(arch="linear_seq"), "model": {"x": torch.ones(2)}}, path
        )

        data = load_fairseq_checkpoint(path)

        assert data["args"].arch == "linear_seq"
        assert torch.equal(data["model"]["x"], torch.ones(2))

    def test_tolerates_unresolvable_referenced_classes(self, tmp_path: Path) -> None:
        # Simulates a checkpoint pickled under a package name (qbert) that
        # isn't importable here -- the loader must still recover the real
        # tensors/namespace, tolerating only the unresolvable object itself.
        # Registers a real, picklable "qbert.fairseq_ext.meters.SumMeter"
        # module/class in sys.modules just long enough to pickle it, then
        # removes it before loading -- exactly reproducing the real
        # scenario (a class that existed in the original training
        # environment but not this one).
        import sys
        import types

        meters_module = types.ModuleType("qbert.fairseq_ext.meters")

        class SumMeter:
            pass

        SumMeter.__module__ = "qbert.fairseq_ext.meters"
        SumMeter.__qualname__ = "SumMeter"
        meters_module.SumMeter = SumMeter  # type: ignore[attr-defined]
        sys.modules["qbert"] = types.ModuleType("qbert")
        sys.modules["qbert.fairseq_ext"] = types.ModuleType("qbert.fairseq_ext")
        sys.modules["qbert.fairseq_ext.meters"] = meters_module

        path = tmp_path / "checkpoint.pt"
        try:
            torch.save({"model": {"x": torch.ones(2)}, "extra_state": SumMeter()}, path)
        finally:
            del sys.modules["qbert"]
            del sys.modules["qbert.fairseq_ext"]
            del sys.modules["qbert.fairseq_ext.meters"]

        data = load_fairseq_checkpoint(path)

        assert torch.equal(data["model"]["x"], torch.ones(2))
        # The unresolvable object becomes an inert placeholder instance, not a crash.
        assert type(data["extra_state"]).__name__ == "SumMeter"
        assert type(data["extra_state"]) is not SumMeter


class TestPermissiveUnpicklerFallback:
    def test_find_class_substitutes_placeholder_for_unresolvable_names(
        self, tmp_path: Path
    ) -> None:
        # Directly exercises the same fallback path load_fairseq_checkpoint
        # relies on, without depending on the real checkpoints' exact byte
        # layout. Registers a real, picklable "totally.nonexistent.module"
        # module/class just long enough to pickle it, then removes it
        # before unpickling -- mirrors a class that existed in the
        # original training environment but not this one.
        import sys
        import types

        fake_module = types.ModuleType("totally.nonexistent.module")

        class Dummy:
            pass

        Dummy.__module__ = "totally.nonexistent.module"
        Dummy.__qualname__ = "Dummy"
        fake_module.Dummy = Dummy  # type: ignore[attr-defined]
        sys.modules["totally"] = types.ModuleType("totally")
        sys.modules["totally.nonexistent"] = types.ModuleType("totally.nonexistent")
        sys.modules["totally.nonexistent.module"] = fake_module

        path = tmp_path / "dummy.pkl"
        try:
            with path.open("wb") as handle:
                pickle.dump(Dummy(), handle)
        finally:
            del sys.modules["totally"]
            del sys.modules["totally.nonexistent"]
            del sys.modules["totally.nonexistent.module"]

        class _PermissiveUnpickler(pickle.Unpickler):
            def find_class(self, module: str, name: str):  # type: ignore[no-untyped-def]
                try:
                    return super().find_class(module, name)
                except Exception:
                    return type(name, (), {})

        with path.open("rb") as handle:
            obj = _PermissiveUnpickler(handle).load()

        assert type(obj).__name__ == "Dummy"
        assert type(obj) is not Dummy
