"""Unit tests for [EwiserLinker][linkingtk.algorithms.wsd.ewiser.EwiserLinker].

Fast/offline-shaped: uses ``hf-internal-testing/tiny-random-BertModel``
(same tiny-model convention as ``test_glossbert.py``/``test_refined.py``)
and a hand-built fake checkpoint file for `from_checkpoint`. Not gated on
a real published EWISER checkpoint (~737MB, user-local) -- that's only
exercised in ``examples/ewiser_reproduction.py``.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from linkingtk.algorithms.wsd._ewiser_decoder import EwiserDecoder  # noqa: E402
from linkingtk.algorithms.wsd._ewiser_vocab import SenseVocabulary  # noqa: E402
from linkingtk.algorithms.wsd.ewiser import EwiserEncoder, EwiserLinker  # noqa: E402
from linkingtk.blocking.exact import ExactMatch  # noqa: E402
from linkingtk.core.entity import Entity  # noqa: E402
from linkingtk.exceptions import LinkingTKError  # noqa: E402

_TINY_MODEL = "hf-internal-testing/tiny-random-BertModel"


def _vocab() -> SenseVocabulary:
    return SenseVocabulary.from_wn(["s-bank-1", "s-bank-2"], nspecial=0)


def _mention_and_senses() -> tuple[Entity, Entity, Entity]:
    mention = Entity(id="m1", labels=["bank"], context=("I sat by the bank today", 13, 17))
    sense1 = Entity(id="s-bank-1", labels=["bank"], description="a financial institution")
    sense2 = Entity(id="s-bank-2", labels=["bank"], description="sloping land beside water")
    return mention, sense1, sense2


class TestEncoderScore:
    def test_score_runs_on_a_pair(self) -> None:
        encoder = EwiserEncoder(
            model_name_or_path=_TINY_MODEL, vocabulary=_vocab(), decoder_hidden_dim=8
        )
        mention, sense1, _sense2 = _mention_and_senses()

        scores = encoder.score([(mention, sense1)])

        assert scores.shape == (1,)

    def test_empty_pairs_returns_empty_tensor(self) -> None:
        encoder = EwiserEncoder(
            model_name_or_path=_TINY_MODEL, vocabulary=_vocab(), decoder_hidden_dim=8
        )

        assert encoder.score([]).shape == (0,)

    def test_mentions_sharing_a_sentence_batch_together(self) -> None:
        # Two mentions from the same sentence share entity1.context[0] --
        # score() must resolve each to its own word position within one
        # shared forward pass, not conflate them.
        encoder = EwiserEncoder(
            model_name_or_path=_TINY_MODEL, vocabulary=_vocab(), decoder_hidden_dim=8
        )
        text = "the bank by the bank"
        near_bank = Entity(id="m1", labels=["bank"], context=(text, 4, 8))
        far_bank = Entity(id="m2", labels=["bank"], context=(text, 17, 21))
        sense1 = Entity(id="s-bank-1", labels=["bank"], description="financial")

        scores = encoder.score([(near_bank, sense1), (far_bank, sense1)])

        assert scores.shape == (2,)

    def test_candidate_not_in_vocabulary_scores_negative_infinity(self) -> None:
        encoder = EwiserEncoder(
            model_name_or_path=_TINY_MODEL, vocabulary=_vocab(), decoder_hidden_dim=8
        )
        mention, _sense1, _sense2 = _mention_and_senses()
        unknown_sense = Entity(id="s-unknown", labels=["bank"], description="not in vocabulary")

        scores = encoder.score([(mention, unknown_sense)])

        assert scores[0].item() == float("-inf")

    def test_requires_a_vocabulary(self) -> None:
        with pytest.raises(LinkingTKError, match="vocabulary"):
            EwiserEncoder(model_name_or_path=_TINY_MODEL)


class TestOutputEmbeddingInit:
    def test_replaces_the_random_default_output_weight(self) -> None:
        vocabulary = _vocab()
        init = torch.arange(len(vocabulary) * 8, dtype=torch.float32).reshape(len(vocabulary), 8)

        encoder = EwiserEncoder(
            model_name_or_path=_TINY_MODEL,
            vocabulary=vocabulary,
            decoder_hidden_dim=8,
            output_embedding_init=init,
        )

        assert torch.equal(encoder.decoder.logits.weight.data, init)

    def test_wrong_shape_raises(self) -> None:
        wrong_shape = torch.zeros(len(_vocab()), 4)  # decoder_hidden_dim=8 below, not 4

        with pytest.raises(LinkingTKError, match="output_embedding_init"):
            EwiserEncoder(
                model_name_or_path=_TINY_MODEL,
                vocabulary=_vocab(),
                decoder_hidden_dim=8,
                output_embedding_init=wrong_shape,
            )


class TestEncoderFreezing:
    def test_freeze_encoder_true_gives_encoder_no_gradients(self) -> None:
        encoder = EwiserEncoder(
            model_name_or_path=_TINY_MODEL,
            vocabulary=_vocab(),
            decoder_hidden_dim=8,
            freeze_encoder=True,
        )
        mention, sense1, _sense2 = _mention_and_senses()

        encoder.score([(mention, sense1)]).sum().backward()

        assert all(p.grad is None for p in encoder.encoder.parameters())
        assert any(p.grad is not None for p in encoder.decoder.parameters())

    def test_freeze_encoder_false_gives_encoder_gradients(self) -> None:
        encoder = EwiserEncoder(
            model_name_or_path=_TINY_MODEL,
            vocabulary=_vocab(),
            decoder_hidden_dim=8,
            freeze_encoder=False,
        )
        mention, sense1, _sense2 = _mention_and_senses()

        encoder.score([(mention, sense1)]).sum().backward()

        assert any(p.grad is not None for p in encoder.encoder.parameters())


class TestLink:
    def test_link_runs_on_untrained_model(self) -> None:
        encoder = EwiserEncoder(
            model_name_or_path=_TINY_MODEL, vocabulary=_vocab(), decoder_hidden_dim=8
        )
        linker = EwiserLinker(encoder)
        mention, sense1, sense2 = _mention_and_senses()

        results = linker.link([mention], [sense1, sense2], blocking=ExactMatch())

        assert {result.source_id for result in results} == {"m1"}

    def test_no_candidates_returns_empty_list(self) -> None:
        encoder = EwiserEncoder(
            model_name_or_path=_TINY_MODEL, vocabulary=_vocab(), decoder_hidden_dim=8
        )
        linker = EwiserLinker(encoder)
        mention = Entity(id="m1", labels=["nonexistent"], context=("no match here", 0, 5))

        results = linker.link([mention], [], blocking=ExactMatch())

        assert results == []


class _FakeSynsetWithIli:
    def __init__(self, id: str, ili: str | None, lemmas: list[str] | None = None) -> None:
        self.id = id
        self.ili = ili
        self._lemmas = lemmas or []

    def definition(self) -> str:
        return ""

    def lemmas(self) -> list[str]:
        return self._lemmas


def _fake_wn_module_with_ili(
    synsets_by_lexicon_and_id: dict[tuple[str, str], _FakeSynsetWithIli],
) -> types.ModuleType:
    module = types.ModuleType("wn")

    def synset(
        id: str, *, lexicon: str | None = None, lang: str | None = None
    ) -> _FakeSynsetWithIli:
        found = synsets_by_lexicon_and_id.get((lexicon, id))
        if found is None:
            raise module.Error(f"no such synset: {id}")  # type: ignore[attr-defined]
        return found

    def synsets(
        form: str | None = None,
        pos: str | None = None,
        ili: str | None = None,
        *,
        lexicon: str | None = None,
        lang: str | None = None,
    ) -> list[_FakeSynsetWithIli]:
        return [
            syn
            for (syn_lexicon, _syn_id), syn in synsets_by_lexicon_and_id.items()
            if syn_lexicon == lexicon
            and (ili is None or syn.ili == ili)
            and (form is None or form in syn._lemmas)
        ]

    module.synset = synset  # type: ignore[attr-defined]
    module.synsets = synsets  # type: ignore[attr-defined]
    module.Error = _FakeWnError  # type: ignore[attr-defined]
    return module


@pytest.fixture
def fake_wn_with_ili(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    # "oewn-00319912-v" and "omw-en-00319111-v" are the real, confirmed
    # cross-lexicon pair from #67's own diagnosis (different byte offsets,
    # same ILI entry "i23324").
    module = _fake_wn_module_with_ili(
        {
            ("oewn:2021", "oewn-00319912-v"): _FakeSynsetWithIli(
                "oewn-00319912-v", ili="i23324", lemmas=["stretch"]
            ),
            ("omw-en:1.4", "omw-en-00319111-v"): _FakeSynsetWithIli(
                "omw-en-00319111-v", ili="i23324"
            ),
            ("oewn:2021", "oewn-no-ili-v"): _FakeSynsetWithIli("oewn-no-ili-v", ili=None),
        }
    )
    monkeypatch.setitem(sys.modules, "wn", module)
    return module


class TestLexiconTranslationForScoring:
    """A `WnEntitySource` built with a different lexicon than the vocabulary's own
    produces ids that never resolve via plain `SenseVocabulary.index_for` --
    `EwiserLinker` translates them via ILI before scoring instead of letting
    every candidate silently score `-inf` (#67)."""

    def _linker(self, vocabulary: SenseVocabulary) -> EwiserLinker:
        encoder = EwiserEncoder(
            model_name_or_path=_TINY_MODEL, vocabulary=vocabulary, decoder_hidden_dim=8
        )
        return EwiserLinker(encoder)

    def test_translates_a_mismatched_candidate_id_via_ili(
        self, fake_wn_with_ili: types.ModuleType
    ) -> None:
        from linkingtk.sources.wn import WnEntitySource

        linker = self._linker(SenseVocabulary(["omw-en-00319111-v"], lexicon="omw-en:1.4"))
        mention, _sense1, _sense2 = _mention_and_senses()
        sense = Entity(id="oewn-00319912-v", labels=["stretch"], description="")

        translated = linker._translate_for_scoring(
            [(mention, sense)], WnEntitySource(lexicon="oewn:2021")
        )

        assert len(translated) == 1
        translated_mention, translated_sense = translated[0]
        assert translated_mention is mention
        assert translated_sense.id == "omw-en-00319111-v"
        assert translated_sense.labels == sense.labels  # everything but id is untouched

    def test_candidate_with_no_ili_match_passes_through_unchanged(
        self, fake_wn_with_ili: types.ModuleType
    ) -> None:
        from linkingtk.sources.wn import WnEntitySource

        linker = self._linker(SenseVocabulary(["omw-en-00319111-v"], lexicon="omw-en:1.4"))
        mention, _sense1, _sense2 = _mention_and_senses()
        sense = Entity(id="oewn-no-ili-v", labels=["x"], description="")

        translated = linker._translate_for_scoring(
            [(mention, sense)], WnEntitySource(lexicon="oewn:2021")
        )

        assert translated == [(mention, sense)]

    def test_matching_lexicon_is_a_no_op(self, fake_wn_with_ili: types.ModuleType) -> None:
        from linkingtk.sources.wn import WnEntitySource

        linker = self._linker(SenseVocabulary(["omw-en-00319111-v"], lexicon="omw-en:1.4"))
        mention, _sense1, _sense2 = _mention_and_senses()
        sense = Entity(id="omw-en-00319111-v", labels=["x"], description="")

        translated = linker._translate_for_scoring(
            [(mention, sense)], WnEntitySource(lexicon="omw-en:1.4")
        )

        assert translated == [(mention, sense)]

    def test_vocabulary_without_a_lexicon_is_a_no_op(
        self, fake_wn_with_ili: types.ModuleType
    ) -> None:
        from linkingtk.sources.wn import WnEntitySource

        linker = self._linker(_vocab())  # from_wn -- no lexicon
        mention, _sense1, _sense2 = _mention_and_senses()
        sense = Entity(id="oewn-00319912-v", labels=["x"], description="")

        translated = linker._translate_for_scoring(
            [(mention, sense)], WnEntitySource(lexicon="oewn:2021")
        )

        assert translated == [(mention, sense)]

    def test_plain_list_dataset2_is_a_no_op(self, fake_wn_with_ili: types.ModuleType) -> None:
        linker = self._linker(SenseVocabulary(["omw-en-00319111-v"], lexicon="omw-en:1.4"))
        mention, _sense1, _sense2 = _mention_and_senses()
        sense = Entity(id="oewn-00319912-v", labels=["x"], description="")

        translated = linker._translate_for_scoring([(mention, sense)], [sense])

        assert translated == [(mention, sense)]

    def test_score_candidates_resolves_across_lexicons_end_to_end(
        self, fake_wn_with_ili: types.ModuleType
    ) -> None:
        from linkingtk.sources.wn import WnEntitySource

        linker = self._linker(SenseVocabulary(["omw-en-00319111-v"], lexicon="omw-en:1.4"))
        mention = Entity(id="m1", labels=["stretch"], context=("stretch the shoe", 0, 7))

        candidates = linker.score_candidates(
            [mention], WnEntitySource(lexicon="oewn:2021", lang="en")
        )

        assert len(candidates["m1"]) == 1
        candidate_id, score = candidates["m1"][0]
        assert candidate_id == "oewn-00319912-v"  # untranslated -- dataset2's own id
        assert score != float("-inf")


class _FakeWnError(Exception):
    pass


class _FakeSynset:
    def __init__(self, id: str) -> None:
        self.id = id


def _fake_wn_module(synsets_by_id: dict[str, _FakeSynset]) -> types.ModuleType:
    module = types.ModuleType("wn")

    def synset(id: str, *, lexicon: str | None = None, lang: str | None = None) -> _FakeSynset:
        found = synsets_by_id.get(id)
        if found is None:
            raise module.Error(f"no such synset: {id}")  # type: ignore[attr-defined]
        return found

    def synsets(
        query: str, *, lexicon: str | None = None, lang: str | None = None
    ) -> list[_FakeSynset]:
        return []  # no candidates -- the lexicon-mismatch tests only care about the check itself

    module.synset = synset  # type: ignore[attr-defined]
    module.synsets = synsets  # type: ignore[attr-defined]
    module.Error = _FakeWnError  # type: ignore[attr-defined]
    return module


@pytest.fixture
def fake_wn(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = _fake_wn_module(
        {
            "omw-en-00000001-n": _FakeSynset("omw-en-00000001-n"),
            "omw-en-00000002-n": _FakeSynset("omw-en-00000002-n"),
        }
    )
    monkeypatch.setitem(sys.modules, "wn", module)
    return module


_TINY_MODEL_HIDDEN_SIZE = 32  # hf-internal-testing/tiny-random-BertModel's own config.hidden_size


def _write_fake_checkpoint(path: Path, vocab_size: int, hidden_dim: int) -> None:
    adjacency = torch.sparse_coo_tensor(
        torch.tensor([[0], [1]]), torch.tensor([0.5]), size=(vocab_size, vocab_size)
    )
    decoder = EwiserDecoder(
        input_dim=_TINY_MODEL_HIDDEN_SIZE,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        adjacency=adjacency,
        structured_logits_trainable=True,
    )
    state_dict = {f"decoder.{key}": value for key, value in decoder.state_dict().items()}
    args = argparse.Namespace(decoder_structured_logits_trainable=True)
    torch.save(
        {"args": args, "model": state_dict, "optimizer_history": [], "extra_state": {}}, path
    )


class TestFromCheckpoint:
    def test_loads_decoder_weights_and_vocabulary(
        self, fake_wn: types.ModuleType, tmp_path: Path
    ) -> None:
        checkpoint_path = tmp_path / "fake.pt"
        _write_fake_checkpoint(checkpoint_path, vocab_size=6, hidden_dim=4)
        offsets_path = tmp_path / "offsets.txt"
        offsets_path.write_text("wn:00000001n 10\nwn:00000002n 5\n")

        encoder = EwiserEncoder.from_checkpoint(
            checkpoint_path, offsets_path, model_name_or_path=_TINY_MODEL
        )

        assert len(encoder.vocabulary) == 6  # 4 reserved + 2 real offsets
        assert encoder.decoder.logits.weight.shape == (6, 4)
        assert encoder.decoder.structured_logits is not None

    def test_mismatched_offsets_file_raises(
        self, fake_wn: types.ModuleType, tmp_path: Path
    ) -> None:
        checkpoint_path = tmp_path / "fake.pt"
        _write_fake_checkpoint(checkpoint_path, vocab_size=6, hidden_dim=4)
        offsets_path = tmp_path / "offsets.txt"
        offsets_path.write_text("wn:00000001n 10\n")  # only 1 entry -> 4+1=5, not 6

        with pytest.raises(LinkingTKError, match="offsets_path"):
            EwiserEncoder.from_checkpoint(
                checkpoint_path, offsets_path, model_name_or_path=_TINY_MODEL
            )

    def test_loaded_encoder_scores_without_error(
        self, fake_wn: types.ModuleType, tmp_path: Path
    ) -> None:
        checkpoint_path = tmp_path / "fake.pt"
        _write_fake_checkpoint(checkpoint_path, vocab_size=6, hidden_dim=4)
        offsets_path = tmp_path / "offsets.txt"
        offsets_path.write_text("wn:00000001n 10\nwn:00000002n 5\n")

        encoder = EwiserEncoder.from_checkpoint(
            checkpoint_path, offsets_path, model_name_or_path=_TINY_MODEL
        )
        mention = Entity(id="m1", labels=["x"], context=("some text here", 0, 4))
        sense = Entity(id="omw-en-00000001-n", labels=["x"], description="")

        scores = encoder.score([(mention, sense)])

        assert scores.shape == (1,)
