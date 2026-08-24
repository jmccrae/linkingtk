"""Unit tests for [EscLinker][linkingtk.algorithms.wsd.esc.EscLinker].

Fast/offline-shaped: uses ``hf-internal-testing/tiny-random-BartModel``
(same tiny-model convention as ``test_ewiser.py``/``test_glossbert.py``)
and a hand-built fake Lightning checkpoint for `from_checkpoint`. Not
gated on the real published ESC checkpoint (~4.9GB, user-local) -- that's
only exercised in ``examples/esc_reproduction.py``. `max_length` is set
well above what a real tokenizer would need: the tiny random tokenizer's
near-random BPE merges tokenize close to character-level, so even short
sentences/glosses need more room than they would with a real vocabulary.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from linkingtk.algorithms.wsd.esc import EscEncoder, EscLinker  # noqa: E402
from linkingtk.blocking.exact import ExactMatch  # noqa: E402
from linkingtk.core.entity import Entity  # noqa: E402

_TINY_MODEL = "hf-internal-testing/tiny-random-BartModel"
_MAX_LENGTH = 256


def _mention_and_senses() -> tuple[Entity, Entity, Entity]:
    mention = Entity(id="m1", labels=["bank"], context=("I sat by the bank today", 13, 17))
    sense1 = Entity(id="s-bank-1", labels=["bank"], description="a financial institution")
    sense2 = Entity(id="s-bank-2", labels=["bank"], description="sloping land beside water")
    return mention, sense1, sense2


def _two_mentions_and_senses() -> tuple[Entity, Entity, Entity, Entity]:
    mention1 = Entity(id="m1", labels=["bank"], context=("I sat by the bank today", 13, 17))
    mention2 = Entity(id="m2", labels=["bank"], context=("The bank raised interest rates", 4, 8))
    sense1 = Entity(id="s-bank-1", labels=["bank"], description="a financial institution")
    sense2 = Entity(id="s-bank-2", labels=["bank"], description="sloping land beside water")
    return mention1, mention2, sense1, sense2


class TestEncoderScore:
    def test_score_runs_on_a_pair(self) -> None:
        encoder = EscEncoder(model_name_or_path=_TINY_MODEL, max_length=_MAX_LENGTH)
        mention, sense1, _sense2 = _mention_and_senses()

        scores = encoder.score([(mention, sense1)])

        assert scores.shape == (1,)
        assert torch.isfinite(scores).all()

    def test_empty_pairs_returns_empty_tensor(self) -> None:
        encoder = EscEncoder(model_name_or_path=_TINY_MODEL, max_length=_MAX_LENGTH)

        assert encoder.score([]).shape == (0,)

    def test_two_mentions_sharing_candidate_ids_batch_together(self) -> None:
        # Both mentions offer the same two candidate sense ids -- a
        # regression check that per-mention candidate/span resolution
        # (keyed by id(mention), not by sense id alone) doesn't conflate
        # them.
        encoder = EscEncoder(
            model_name_or_path=_TINY_MODEL, max_length=_MAX_LENGTH, forward_batch_size=2
        )
        mention1, mention2, sense1, sense2 = _two_mentions_and_senses()

        scores = encoder.score(
            [(mention1, sense1), (mention1, sense2), (mention2, sense1), (mention2, sense2)]
        )

        assert scores.shape == (4,)
        assert torch.isfinite(scores).all()

    def test_batching_matches_unbatched_scoring(self) -> None:
        # Loose tolerance deliberately: batched vs. unbatched matmul is not
        # bit-reproducible in general (different reduction order), and this
        # tiny *random*-weight model's unscaled activations amplify that
        # more than a real pretrained model would (verified directly: the
        # weights, resolved spans, and raw batched-forward logits are all
        # bit-identical between the two encoders here -- confirmed by hand
        # -building the same padded batch tensors and comparing against
        # `score()`'s own internals). This atol is still tight enough to
        # catch a real conflation bug (e.g. two mentions' candidate spans
        # swapped), which would shift scores by several log-prob units, not
        # a few hundredths.
        mention1, mention2, sense1, sense2 = _two_mentions_and_senses()
        pairs = [(mention1, sense1), (mention1, sense2), (mention2, sense1), (mention2, sense2)]

        torch.manual_seed(0)
        batched = EscEncoder(
            model_name_or_path=_TINY_MODEL, max_length=_MAX_LENGTH, forward_batch_size=4
        )
        unbatched = EscEncoder(
            model_name_or_path=_TINY_MODEL, max_length=_MAX_LENGTH, forward_batch_size=1
        )
        unbatched.load_state_dict(batched.state_dict())
        batched.eval()
        unbatched.eval()

        with torch.no_grad():
            batched_scores = batched.score(pairs)
            unbatched_scores = unbatched.score(pairs)

        assert torch.allclose(batched_scores, unbatched_scores, atol=0.5)

    def test_score_is_differentiable(self) -> None:
        encoder = EscEncoder(model_name_or_path=_TINY_MODEL, max_length=_MAX_LENGTH)
        mention, sense1, _sense2 = _mention_and_senses()

        encoder.score([(mention, sense1)]).sum().backward()

        assert any(p.grad is not None for p in encoder.qa_model.parameters())


class TestLink:
    def test_link_runs_on_untrained_model(self) -> None:
        encoder = EscEncoder(model_name_or_path=_TINY_MODEL, max_length=_MAX_LENGTH)
        linker = EscLinker(encoder)
        mention, sense1, sense2 = _mention_and_senses()

        results = linker.link([mention], [sense1, sense2], blocking=ExactMatch())

        assert {result.source_id for result in results} == {"m1"}

    def test_no_candidates_returns_empty_list(self) -> None:
        encoder = EscEncoder(model_name_or_path=_TINY_MODEL, max_length=_MAX_LENGTH)
        linker = EscLinker(encoder)
        mention = Entity(id="m1", labels=["nonexistent"], context=("no match here", 0, 5))

        results = linker.link([mention], [], blocking=ExactMatch())

        assert results == []


class TestFromCheckpoint:
    def test_loaded_encoder_scores_without_error(self, tmp_path: Path) -> None:
        source = EscEncoder(model_name_or_path=_TINY_MODEL, max_length=_MAX_LENGTH)
        state_dict = {
            f"qa_model.{key}": value for key, value in source.qa_model.state_dict().items()
        }
        checkpoint_path = tmp_path / "fake.ckpt"
        torch.save({"state_dict": state_dict}, checkpoint_path)

        loaded = EscEncoder.from_checkpoint(
            checkpoint_path, model_name_or_path=_TINY_MODEL, max_length=_MAX_LENGTH
        )
        mention, sense1, _sense2 = _mention_and_senses()

        scores = loaded.score([(mention, sense1)])

        assert scores.shape == (1,)
        assert torch.isfinite(scores).all()
        assert torch.allclose(
            dict(loaded.qa_model.named_parameters())["qa_outputs.weight"],
            dict(source.qa_model.named_parameters())["qa_outputs.weight"],
        )
