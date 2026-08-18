"""Unit tests for [Trainer][linkingtk.train.trainer.Trainer].

No bi-encoder/GNN linker in this repo implements
[TrainableModel][linkingtk.train.trainer.TrainableModel] yet -- these
tests validate ``Trainer`` against a small in-repo toy encoder instead: a
per-entity embedding table (no coupling between a KG1 entity and its true
KG2 match at initialization, so untrained Hits@1 starts near chance) plus
a named ``nn.Linear`` projection, so PEFT's ``target_modules`` has a real
layer to adapt. Same "transductive embedding table" shape as
``BootEALinker``/``SEALinker``'s own hand-rolled training loops (see
test_sea.py's "recovers seeded alignment" test) -- a pipeline-correctness
check, not a generalization benchmark.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from peft import LoraConfig  # noqa: E402
from torch import nn  # noqa: E402

from linkingtk.blocking.label_overlap import LabelOverlap  # noqa: E402
from linkingtk.core.entity import Entity, label_texts  # noqa: E402
from linkingtk.exceptions import LinkingTKError  # noqa: E402
from linkingtk.train.arguments import TrainingArguments  # noqa: E402
from linkingtk.train.trainer import Trainer  # noqa: E402

# Greek-letter names share many character n-grams with each other (e.g.
# "theta"/"eta"/"zeta"), so LabelOverlap surfaces real false candidates to
# mine as hard negatives -- no artificial filler text needed.
_NAMES = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa"]


def _toy_ea_pairs() -> list[tuple[Entity, Entity]]:
    kg1 = [Entity(id=f"kg1:{name}", labels=[name]) for name in _NAMES]
    kg2 = [Entity(id=f"kg2:{name}", labels=[name]) for name in _NAMES]
    return list(zip(kg1, kg2, strict=True))


class _ToyEncoder(nn.Module):
    """Per-entity embedding table (keyed by id, not label) + a named Linear projection.

    Keying by id -- rather than by label text, which would let a KG1
    entity and its true KG2 match with the same name share an embedding
    row before any training happens -- guarantees untrained cosine
    similarity between true pairs starts uncorrelated, so a Hits@1
    improvement after training is real signal, not initialization leakage.
    """

    def __init__(
        self, entity_ids: list[str], embedding_dim: int = 16, random_state: int = 0
    ) -> None:
        super().__init__()
        # Seeds the *global* torch RNG, not just a local Generator -- nn.Linear's
        # default reset_parameters() (used by self.proj below) draws from the
        # global RNG, so a local-only Generator would leave self.proj's initial
        # weights (and therefore every downstream result) dependent on whatever
        # ambient global RNG state other tests left behind.
        torch.manual_seed(random_state)
        self._index = {entity_id: i for i, entity_id in enumerate(entity_ids)}
        self.embedding = nn.Embedding(len(entity_ids), embedding_dim)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.randn(len(entity_ids), embedding_dim))
        self.proj = nn.Linear(embedding_dim, embedding_dim)

    def encode(self, entities: list[Entity]) -> torch.Tensor:
        device = self.proj.weight.device
        ids = torch.tensor([self._index[entity.id] for entity in entities], device=device)
        return torch.nn.functional.normalize(self.proj(self.embedding(ids)), dim=-1)


def _make_trainer(loss: str, **kwargs: object) -> tuple[Trainer, _ToyEncoder]:
    pairs = _toy_ea_pairs()
    entity_ids = [e.id for pair in pairs for e in pair]
    model = _ToyEncoder(entity_ids)
    defaults: dict[str, object] = {
        "output_dir": str(kwargs.pop("output_dir")),
        "learning_rate": 0.1,
        "num_epochs": 300,
        "batch_size": 32,
        "negative_samples_ratio": 3,
        "loss": loss,
        "margin": 0.3,
    }
    defaults.update(kwargs)
    args = TrainingArguments(**defaults)  # type: ignore[arg-type]
    trainer = Trainer(
        model=model,
        args=args,
        train_data=pairs,
        eval_data=pairs,
        blocking=LabelOverlap(ngram_size=2, max_matches=5),
    )
    return trainer, model


def _hits_at_1(trainer: Trainer, model: _ToyEncoder) -> float:
    dataset1 = [e1 for e1, _ in trainer.train_data]
    dataset2 = [e2 for _, e2 in trainer.train_data]
    ground_truth = [(e1.id, e2.id) for e1, e2 in trainer.train_data]
    with torch.no_grad():
        emb1 = model.encode(dataset1)
        emb2 = model.encode(dataset2)
    similarities = emb1 @ emb2.T
    correct = 0
    for i, (_source_id, target_id) in enumerate(ground_truth):
        predicted = dataset2[int(torch.argmax(similarities[i]))].id
        correct += predicted == target_id
    return correct / len(ground_truth)


class TestTrainImprovesAlignment:
    def test_infonce_reaches_perfect_hits_at_1(self, tmp_path: Path) -> None:
        # InfoNCE's negatives are every other in-batch positive (all 9 other
        # entities here, batch_size=32 > 10 pairs) plus mined hard negatives --
        # full coverage of the 10-way disambiguation, so it should fully resolve.
        trainer, model = _make_trainer("infonce", output_dir=tmp_path / "model")
        before = _hits_at_1(trainer, model)

        trainer.train()

        after = _hits_at_1(trainer, model)
        assert after > before
        assert after == 1.0

    def test_margin_loss_improves_hits_at_1(self, tmp_path: Path) -> None:
        # Margin loss only sees each anchor's mined hard negatives (top-3 of
        # LabelOverlap's top-5 confusable neighbors here), not every other
        # entity -- less negative coverage than InfoNCE's in-batch negatives,
        # so it reliably improves but doesn't reach the same perfect
        # disambiguation in this toy setup -- needs more epochs too, since
        # each step's signal is sparser (3 negatives vs. every batch-mate).
        trainer, model = _make_trainer("margin", output_dir=tmp_path / "model", num_epochs=600)
        before = _hits_at_1(trainer, model)

        trainer.train()

        after = _hits_at_1(trainer, model)
        assert after > before
        assert after >= 0.5

    def test_eval_history_populated_and_improving(self, tmp_path: Path) -> None:
        trainer, _model = _make_trainer("infonce", output_dir=tmp_path / "model")

        trainer.train()

        assert len(trainer.eval_history) == trainer.args.num_epochs
        assert (
            trainer.eval_history[-1].metrics["Hits@1"] >= trainer.eval_history[0].metrics["Hits@1"]
        )
        assert trainer.eval_history[-1].metrics["Hits@1"] == 1.0


class TestCheckpoint:
    def test_saves_model_state_dict(self, tmp_path: Path) -> None:
        trainer, _model = _make_trainer("infonce", output_dir=tmp_path / "model")

        trainer.train()

        assert (tmp_path / "model" / "model.pt").exists()


class TestPeft:
    def test_use_peft_without_config_raises(self, tmp_path: Path) -> None:
        trainer, _model = _make_trainer("infonce", output_dir=tmp_path / "model", use_peft=True)

        with pytest.raises(LinkingTKError, match="peft_config"):
            trainer.train()

    def test_use_peft_with_config_reduces_trainable_params_and_trains(self, tmp_path: Path) -> None:
        trainer, model = _make_trainer(
            "infonce",
            output_dir=tmp_path / "model",
            use_peft=True,
            peft_config=LoraConfig(target_modules=["proj"], r=4, lora_alpha=8),
        )
        full_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        trainer.train()

        peft_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        assert peft_params < full_params
        assert trainer.eval_history[-1].metrics["Hits@1"] == 1.0


def test_label_texts_used_for_blocking_not_model_identity() -> None:
    # Sanity check on the fixture itself: LabelOverlap indeed finds
    # cross-name false candidates (the hard negatives Trainer mines),
    # confirming _NAMES' shared substrings do what the module docstring claims.
    pairs = _toy_ea_pairs()
    kg1 = [e1 for e1, _ in pairs]
    kg2 = [e2 for _, e2 in pairs]
    blocking = LabelOverlap(ngram_size=2, max_matches=5)
    candidates = blocking.candidate_pairs(kg1, kg2)
    ground_truth = {(e1.id, e2.id) for e1, e2 in pairs}
    false_candidates = [
        (e1.id, e2.id) for e1, e2 in candidates if (e1.id, e2.id) not in ground_truth
    ]
    assert false_candidates
    assert all(label_texts(e1) and label_texts(e2) for e1, e2 in candidates)
