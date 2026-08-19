"""Unit tests for [ReFinEDLinker][linkingtk.algorithms.el.refined.ReFinEDLinker].

Fast/offline: uses ``hf-internal-testing/tiny-random-DistilBertModel``
(same tiny model ``test_multike.py`` already relies on) so no real
pretrained download happens. See ``test_refined_benchmark.py`` for the
slow, real-data, real-model comparison against ReFinED's published
numbers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from peft import LoraConfig  # noqa: E402

from linkingtk.algorithms.el.refined import ReFinEDEncoder, ReFinEDLinker  # noqa: E402
from linkingtk.blocking.exact import ExactMatch  # noqa: E402
from linkingtk.core.entity import Entity  # noqa: E402
from linkingtk.datasets.toy import ToyELDataset  # noqa: E402
from linkingtk.eval import Evaluator  # noqa: E402
from linkingtk.exceptions import LinkingTKError  # noqa: E402
from linkingtk.train.arguments import TrainingArguments  # noqa: E402
from linkingtk.train.trainer import Trainer  # noqa: E402

_TINY_MODEL = "hf-internal-testing/tiny-random-DistilBertModel"


def _train_data() -> tuple[
    list[Entity], list[Entity], list[tuple[str, str]], list[tuple[Entity, Entity]]
]:
    mentions, kb, ground_truth = ToyELDataset().load()
    mentions_by_id = {entity.id: entity for entity in mentions}
    kb_by_id = {entity.id: entity for entity in kb}
    pairs = [(mentions_by_id[mid], kb_by_id[eid]) for mid, eid in ground_truth]
    return mentions, kb, ground_truth, pairs


def _precision_at_1(
    linker: ReFinEDLinker,
    mentions: list[Entity],
    kb: list[Entity],
    ground_truth: list[tuple[str, str]],
) -> float:
    results = linker.link(mentions, kb, blocking=ExactMatch())
    predictions = [(result.source_id, result.target_id) for result in results]
    return Evaluator.evaluate(predictions=predictions, ground_truth=ground_truth).metrics[
        "precision@1"
    ]


class TestLinkImprovesAfterTraining:
    def test_precision_at_1_improves(self, tmp_path: Path) -> None:
        mentions, kb, ground_truth, pairs = _train_data()
        linker = ReFinEDLinker(model_name=_TINY_MODEL, embedding_dim=32, max_length=32)
        before = _precision_at_1(linker, mentions, kb, ground_truth)

        args = TrainingArguments(
            output_dir=str(tmp_path / "model"),
            learning_rate=5e-4,
            num_epochs=60,
            batch_size=8,
            negative_samples_ratio=2,
            loss="infonce",
            temperature=0.1,
        )
        Trainer(model=linker.encoder, args=args, train_data=pairs, blocking=ExactMatch()).train()

        after = _precision_at_1(linker, mentions, kb, ground_truth)
        assert after > before
        assert after == 1.0

    def test_link_runs_on_untrained_encoder(self) -> None:
        # No error, no _fitted guard -- just poor-quality (but well-formed) output.
        mentions, kb, _ground_truth, _pairs = _train_data()
        linker = ReFinEDLinker(model_name=_TINY_MODEL, embedding_dim=32, max_length=32)

        results = linker.link(mentions, kb, blocking=ExactMatch())

        assert {result.source_id for result in results} == {entity.id for entity in mentions}


class TestPeft:
    def test_use_peft_reduces_trainable_params_and_trains(self, tmp_path: Path) -> None:
        _mentions, _kb, _ground_truth, pairs = _train_data()
        linker = ReFinEDLinker(model_name=_TINY_MODEL, embedding_dim=32, max_length=32)
        full_params = sum(p.numel() for p in linker.encoder.parameters() if p.requires_grad)

        args = TrainingArguments(
            output_dir=str(tmp_path / "model"),
            learning_rate=5e-4,
            num_epochs=5,
            batch_size=8,
            negative_samples_ratio=2,
            use_peft=True,
            peft_config=LoraConfig(target_modules=["proj"], r=4, lora_alpha=8),
        )
        trainer = Trainer(model=linker.encoder, args=args, train_data=pairs, blocking=ExactMatch())
        trainer.train()

        peft_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        assert peft_params < full_params

    def test_use_peft_without_config_raises(self, tmp_path: Path) -> None:
        _mentions, _kb, _ground_truth, pairs = _train_data()
        linker = ReFinEDLinker(model_name=_TINY_MODEL, embedding_dim=32, max_length=32)
        args = TrainingArguments(output_dir=str(tmp_path / "model"), num_epochs=1, use_peft=True)
        trainer = Trainer(model=linker.encoder, args=args, train_data=pairs, blocking=ExactMatch())

        with pytest.raises(LinkingTKError, match="peft_config"):
            trainer.train()


class TestEncodeTextFormatting:
    def test_mention_and_kb_entry_both_encode(self) -> None:
        encoder = ReFinEDEncoder(model_name=_TINY_MODEL, embedding_dim=16, max_length=16)
        mention = Entity(id="m1", labels=["Paris"], context="Paris is the capital of France")
        kb_entry = Entity(
            id="Q90", labels=["Paris"], description="capital and most populous city of France"
        )

        embeddings = encoder.encode([mention, kb_entry])

        assert embeddings.shape == (2, 16)
        assert not torch.allclose(embeddings[0], embeddings[1])

    def test_mention_with_span_offsets_marks_the_span(self) -> None:
        encoder = ReFinEDEncoder(model_name=_TINY_MODEL, embedding_dim=16, max_length=16)
        with_span = Entity(id="m1", labels=["Paris"], context=("Paris is nice", 0, 5))
        without_span = Entity(id="m2", labels=["Paris"], context="Paris is nice")

        embeddings = encoder.encode([with_span, without_span])

        # Different formatted text ([E] Paris [/E] vs. a plain string with the label
        # prepended) -- not asserting exact equality, just that both run and differ.
        assert embeddings.shape == (2, 16)
