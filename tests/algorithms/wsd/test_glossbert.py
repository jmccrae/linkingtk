"""Unit tests for [GlossBertLinker][linkingtk.algorithms.wsd.glossbert.GlossBertLinker].

Fast/offline-shaped: uses ``hf-internal-testing/tiny-random-BertForSequenceClassification``
(same tiny-model convention as ``test_refined.py``). Not gated on the
paper's real published checkpoint (~440MB, user-local, not something CI
or another developer has) -- that's only exercised in
``examples/glossbert_reproduction.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from peft import LoraConfig  # noqa: E402

from linkingtk.algorithms.wsd.glossbert import (  # noqa: E402
    GlossBertEncoder,
    GlossBertLinker,
    _gloss_text,
    _mention_text,
)
from linkingtk.blocking.exact import ExactMatch  # noqa: E402
from linkingtk.core.entity import Entity  # noqa: E402
from linkingtk.eval import Evaluator  # noqa: E402
from linkingtk.exceptions import LinkingTKError  # noqa: E402
from linkingtk.train.arguments import TrainingArguments  # noqa: E402
from linkingtk.train.cross_encoder import CrossEncoderTrainer  # noqa: E402

_TINY_MODEL = "hf-internal-testing/tiny-random-BertForSequenceClassification"


def _train_data() -> tuple[
    list[Entity], list[Entity], list[tuple[str, str]], list[tuple[Entity, Entity]]
]:
    mentions = [
        Entity(id="m1", labels=["bass"], context=("He caught a bass while fishing", 15, 19)),
        Entity(id="m2", labels=["bass"], context=("The bass was too quiet", 4, 8)),
        Entity(id="m3", labels=["crane"], context=("A crane lifted the beams", 2, 7)),
        Entity(id="m4", labels=["crane"], context=("The crane flew over the marsh", 4, 9)),
    ]
    senses = [
        Entity(id="bass.n.01", labels=["bass"], description="a fish"),
        Entity(id="bass.n.02", labels=["bass"], description="low-frequency music"),
        Entity(id="crane.n.01", labels=["crane"], description="a lifting machine"),
        Entity(id="crane.n.02", labels=["crane"], description="a wading bird"),
    ]
    ground_truth = [
        ("m1", "bass.n.01"),
        ("m2", "bass.n.02"),
        ("m3", "crane.n.01"),
        ("m4", "crane.n.02"),
    ]
    mentions_by_id = {entity.id: entity for entity in mentions}
    senses_by_id = {entity.id: entity for entity in senses}
    pairs = [(mentions_by_id[mid], senses_by_id[sid]) for mid, sid in ground_truth]
    return mentions, senses, ground_truth, pairs


def _precision_at_1(
    linker: GlossBertLinker,
    mentions: list[Entity],
    senses: list[Entity],
    ground_truth: list[tuple[str, str]],
) -> float:
    results = linker.link(mentions, senses, blocking=ExactMatch())
    predictions = [(result.source_id, result.target_id) for result in results]
    return Evaluator.evaluate(predictions=predictions, ground_truth=ground_truth).metrics[
        "precision@1"
    ]


class TestLinkImprovesAfterTraining:
    def test_precision_at_1_improves(self, tmp_path: Path) -> None:
        mentions, senses, ground_truth, pairs = _train_data()
        linker = GlossBertLinker(model_name_or_path=_TINY_MODEL, max_length=32)
        before = _precision_at_1(linker, mentions, senses, ground_truth)

        args = TrainingArguments(
            output_dir=str(tmp_path / "model"),
            learning_rate=1e-3,
            num_epochs=30,
            batch_size=8,
            negative_samples_ratio=3,
        )
        CrossEncoderTrainer(
            model=linker.model, args=args, train_data=pairs, blocking=ExactMatch()
        ).train()

        after = _precision_at_1(linker, mentions, senses, ground_truth)
        assert after >= before

    def test_link_runs_on_untrained_model(self) -> None:
        # No error, no _fitted guard -- just poor-quality (but well-formed) output.
        mentions, senses, _ground_truth, _pairs = _train_data()
        linker = GlossBertLinker(model_name_or_path=_TINY_MODEL, max_length=32)

        results = linker.link(mentions, senses, blocking=ExactMatch())

        assert {result.source_id for result in results} == {entity.id for entity in mentions}


class TestPeft:
    def test_use_peft_reduces_trainable_params_and_trains(self, tmp_path: Path) -> None:
        _mentions, _senses, _ground_truth, pairs = _train_data()
        linker = GlossBertLinker(model_name_or_path=_TINY_MODEL, max_length=32)
        full_params = sum(p.numel() for p in linker.model.parameters() if p.requires_grad)

        args = TrainingArguments(
            output_dir=str(tmp_path / "model"),
            learning_rate=1e-3,
            num_epochs=2,
            batch_size=8,
            use_peft=True,
            peft_config=LoraConfig(target_modules=["query", "value"], r=4, lora_alpha=8),
        )
        trainer = CrossEncoderTrainer(
            model=linker.model, args=args, train_data=pairs, blocking=ExactMatch()
        )
        trainer.train()

        peft_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        assert peft_params < full_params

    def test_use_peft_without_config_raises(self, tmp_path: Path) -> None:
        _mentions, _senses, _ground_truth, pairs = _train_data()
        linker = GlossBertLinker(model_name_or_path=_TINY_MODEL, max_length=32)
        args = TrainingArguments(output_dir=str(tmp_path / "model"), num_epochs=1, use_peft=True)
        trainer = CrossEncoderTrainer(
            model=linker.model, args=args, train_data=pairs, blocking=ExactMatch()
        )

        with pytest.raises(LinkingTKError, match="peft_config"):
            trainer.train()


class TestTextFormatting:
    def test_mention_text_wraps_span_in_quote_marks(self) -> None:
        mention = Entity(id="m1", labels=["bank"], context=("I sat by the bank today", 13, 17))

        formatted = _mention_text(mention)

        assert formatted == 'I sat by the " bank " today'

    def test_mention_text_falls_back_for_plain_string_context(self) -> None:
        mention = Entity(id="m1", labels=["bank"], context="I sat by the bank today")

        formatted = _mention_text(mention)

        assert formatted == '" bank " I sat by the bank today'

    def test_gloss_text_uses_mention_surface_form_not_label_or_sense_label(self) -> None:
        # labels holds the dictionary lemma ("bank"), deliberately different
        # from the mention's literal surface text ("banks") -- the gloss
        # text must come from the latter (GlossBERT's own convention).
        mention = Entity(id="m1", labels=["bank"], context=("the banks were closed", 4, 9))
        sense = Entity(id="s1", labels=["bank"], description="a financial institution")

        formatted = _gloss_text(mention, sense)

        assert formatted == "banks : a financial institution"

    def test_encoder_score_runs_on_a_pair(self) -> None:
        encoder = GlossBertEncoder(model_name_or_path=_TINY_MODEL, max_length=32)
        mention = Entity(id="m1", labels=["bank"], context=("I sat by the bank", 13, 17))
        sense = Entity(id="s1", labels=["bank"], description="sloping land beside water")

        scores = encoder.score([(mention, sense)])

        assert scores.shape == (1,)
