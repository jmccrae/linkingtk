"""Unit tests for [CrossEncoderTrainer][linkingtk.train.cross_encoder.CrossEncoderTrainer].

[GlossBertLinker][linkingtk.algorithms.wsd.glossbert.GlossBertLinker] (see
`test_glossbert.py`) is the only real
[CrossEncoderModel][linkingtk.train.cross_encoder.CrossEncoderModel] in
this repo -- these tests validate the trainer itself against a small
in-repo toy scorer instead, same "toy encoder, real trainer" split as
`test_trainer.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from peft import LoraConfig  # noqa: E402
from torch import nn  # noqa: E402

from linkingtk.blocking.exact import ExactMatch  # noqa: E402
from linkingtk.core.entity import Entity  # noqa: E402
from linkingtk.core.source import EntitySource  # noqa: E402
from linkingtk.exceptions import LinkingTKError  # noqa: E402
from linkingtk.train.arguments import TrainingArguments  # noqa: E402
from linkingtk.train.cross_encoder import CrossEncoderTrainer  # noqa: E402

# Distinct id spaces per side, disjoint from every other word's ids, so a
# hard negative (a different word's same-prefix "wX" label under
# ExactMatch) starts scoring plausibly before training.
_WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]


def _toy_pairs() -> list[tuple[Entity, Entity]]:
    mentions = [Entity(id=f"m:{w}", labels=[w]) for w in _WORDS]
    senses = [Entity(id=f"s:{w}", labels=[w]) for w in _WORDS]
    return list(zip(mentions, senses, strict=True))


class _ToyScorer(nn.Module):
    """Per-(mention word, sense word) pair lookup table + a named Linear layer.

    Keyed by each side's *word*, not by whether the pair is a true match --
    the model has to learn which same-word pairs are the real match purely
    from training signal, same "no initialization leakage" property
    `test_trainer.py::_ToyEncoder` documents.
    """

    def __init__(self, words: list[str], embedding_dim: int = 8, random_state: int = 0) -> None:
        super().__init__()
        torch.manual_seed(random_state)
        self._index = {word: i for i, word in enumerate(words)}
        self.mention_embedding = nn.Embedding(len(words), embedding_dim)
        self.sense_embedding = nn.Embedding(len(words), embedding_dim)
        with torch.no_grad():
            self.mention_embedding.weight.copy_(torch.randn(len(words), embedding_dim))
            self.sense_embedding.weight.copy_(torch.randn(len(words), embedding_dim))
        self.proj = nn.Linear(embedding_dim, 1)

    def score(self, pairs: list[tuple[Entity, Entity]]) -> torch.Tensor:
        device = self.proj.weight.device
        mention_ids = torch.tensor(
            [self._index[entity1.labels[0]] for entity1, _entity2 in pairs], device=device
        )
        sense_ids = torch.tensor(
            [self._index[entity2.labels[0]] for _entity1, entity2 in pairs], device=device
        )
        joint = self.mention_embedding(mention_ids) * self.sense_embedding(sense_ids)
        return self.proj(joint).squeeze(-1)


def _make_trainer(**kwargs: object) -> tuple[CrossEncoderTrainer, _ToyScorer]:
    pairs = _toy_pairs()
    model = _ToyScorer(_WORDS)
    defaults: dict[str, object] = {
        "output_dir": str(kwargs.pop("output_dir")),
        "learning_rate": 0.1,
        "num_epochs": 100,
        "batch_size": 16,
        "negative_samples_ratio": 3,
    }
    defaults.update(kwargs)
    args = TrainingArguments(**defaults)  # type: ignore[arg-type]
    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_data=pairs,
        eval_data=pairs,
        blocking=ExactMatch(),
    )
    return trainer, model


def _precision_at_1(trainer: CrossEncoderTrainer, model: _ToyScorer) -> float:
    senses = [e2 for _, e2 in trainer.train_data]
    with torch.no_grad():
        correct = 0
        for mention, true_sense in trainer.train_data:
            pairs = [(mention, sense) for sense in senses]
            scores = model.score(pairs)
            predicted = senses[int(torch.argmax(scores))].id
            correct += predicted == true_sense.id
    return correct / len(trainer.train_data)


class TestTrainImprovesPrecision:
    def test_precision_at_1_improves(self, tmp_path: Path) -> None:
        trainer, model = _make_trainer(output_dir=tmp_path / "model")
        before = _precision_at_1(trainer, model)

        trainer.train()

        after = _precision_at_1(trainer, model)
        assert after > before
        assert after == 1.0

    def test_eval_history_populated_and_improving(self, tmp_path: Path) -> None:
        trainer, _model = _make_trainer(output_dir=tmp_path / "model")

        trainer.train()

        assert len(trainer.eval_history) == trainer.args.num_epochs
        assert (
            trainer.eval_history[-1].metrics["Hits@1"] >= trainer.eval_history[0].metrics["Hits@1"]
        )
        assert trainer.eval_history[-1].metrics["Hits@1"] == 1.0


class _SpyBlocking(ExactMatch):
    """Records every `dataset2` it's called with, alongside real `ExactMatch` behavior."""

    def __init__(self) -> None:
        super().__init__()
        self.dataset2_seen: list[list[Entity] | EntitySource] = []

    def candidate_pairs(
        self, dataset1: list[Entity], dataset2: list[Entity] | EntitySource
    ) -> list[tuple[Entity, Entity]]:
        self.dataset2_seen.append(dataset2)
        return super().candidate_pairs(dataset1, dataset2)


class TestEvalDataset2:
    """Regression coverage: `_evaluate()` used to always derive its candidate
    pool from `eval_data`'s own (few, already-correct) targets, silently
    scoring the WSD task far too easy -- measured directly on real SemCor
    data (issue #39's benchmark investigation): 60% self-reported Hits@1
    vs. 33% via the real `GlossBertLinker.link()` path on the same
    held-out mentions and trained model.
    """

    def test_defaults_to_derived_dataset2_from_eval_data(self, tmp_path: Path) -> None:
        pairs = _toy_pairs()
        trainer, _model = _make_trainer(output_dir=tmp_path / "model", num_epochs=1)
        spy = _SpyBlocking()
        trainer.blocking = spy

        trainer.train()

        eval_dataset2 = spy.dataset2_seen[-1]
        assert isinstance(eval_dataset2, list)
        assert {e.id for e in eval_dataset2} == {e2.id for _e1, e2 in pairs}

    def test_explicit_eval_dataset2_is_used_instead(self, tmp_path: Path) -> None:
        pairs = _toy_pairs()
        model = _ToyScorer(_WORDS)
        args = TrainingArguments(
            output_dir=str(tmp_path / "model"),
            learning_rate=0.1,
            num_epochs=1,
            batch_size=16,
            negative_samples_ratio=3,
        )
        wider_dataset2 = [e2 for _e1, e2 in pairs] + [Entity(id="s:decoy", labels=["alpha"])]
        spy = _SpyBlocking()
        trainer = CrossEncoderTrainer(
            model=model,
            args=args,
            train_data=pairs,
            eval_data=pairs,
            eval_dataset2=wider_dataset2,
            blocking=spy,
        )

        trainer.train()

        eval_dataset2 = spy.dataset2_seen[-1]
        assert isinstance(eval_dataset2, list)
        assert {e.id for e in eval_dataset2} == {e.id for e in wider_dataset2}


class TestCheckpoint:
    def test_saves_model_state_dict(self, tmp_path: Path) -> None:
        trainer, _model = _make_trainer(output_dir=tmp_path / "model")

        trainer.train()

        assert (tmp_path / "model" / "model.pt").exists()

    def test_saves_after_every_epoch_not_just_at_the_end(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A long unattended run can be interrupted mid-training -- the
        # checkpoint must be refreshed every epoch, not only once train()
        # fully returns, or an interruption loses all prior progress.
        trainer, _model = _make_trainer(output_dir=tmp_path / "model", num_epochs=5)
        save_calls = []
        real_save = torch.save
        monkeypatch.setattr(
            torch, "save", lambda obj, path: (save_calls.append(path), real_save(obj, path))
        )

        trainer.train()

        assert len(save_calls) == 5


class TestPeft:
    def test_use_peft_without_config_raises(self, tmp_path: Path) -> None:
        trainer, _model = _make_trainer(output_dir=tmp_path / "model", use_peft=True)

        with pytest.raises(LinkingTKError, match="peft_config"):
            trainer.train()

    def test_use_peft_with_config_reduces_trainable_params_and_trains(self, tmp_path: Path) -> None:
        trainer, model = _make_trainer(
            output_dir=tmp_path / "model",
            use_peft=True,
            peft_config=LoraConfig(target_modules=["proj"], r=4, lora_alpha=8),
        )
        full_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        trainer.train()

        peft_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        assert peft_params < full_params
