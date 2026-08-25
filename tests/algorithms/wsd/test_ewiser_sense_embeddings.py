from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from linkingtk.algorithms.wsd._ewiser_sense_embeddings import (  # noqa: E402
    build_synset_centroid_vectors_from_lmms,
    load_synset_centroid_vectors,
)
from linkingtk.algorithms.wsd._ewiser_vocab import SenseVocabulary  # noqa: E402


class _FakeWnError(Exception):
    pass


class _FakeSynset:
    def __init__(self, id: str) -> None:
        self.id = id


class _FakeSense:
    def __init__(self, identifier: str, synset: _FakeSynset) -> None:
        self._identifier = identifier
        self._synset = synset

    def metadata(self) -> dict[str, str]:
        return {"identifier": self._identifier}

    def synset(self) -> _FakeSynset:
        return self._synset


def _fake_wn_module(
    synsets_by_id: dict[str, _FakeSynset],
    senses_by_lemma: dict[str, list[_FakeSense]] | None = None,
) -> types.ModuleType:
    module = types.ModuleType("wn")

    def synset(id: str, *, lexicon: str | None = None, lang: str | None = None) -> _FakeSynset:
        found = synsets_by_id.get(id)
        if found is None:
            raise module.Error(f"no such synset: {id}")  # type: ignore[attr-defined]
        return found

    def senses(
        form: str | None = None, pos: str | None = None, *, lexicon: str | None = None
    ) -> list[_FakeSense]:
        return (senses_by_lemma or {}).get(form or "", [])

    module.synset = synset  # type: ignore[attr-defined]
    module.senses = senses  # type: ignore[attr-defined]
    module.Error = _FakeWnError  # type: ignore[attr-defined]
    return module


class TestLoadSynsetCentroidVectors:
    def test_fills_matching_rows_and_leaves_the_rest_untouched(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setitem(
            sys.modules,
            "wn",
            _fake_wn_module(
                {
                    "omw-en-02084071-n": _FakeSynset("omw-en-02084071-n"),
                    "omw-en-00007846-n": _FakeSynset("omw-en-00007846-n"),
                }
            ),
        )
        vectors_file = tmp_path / "vectors.vec"
        vectors_file.write_text(
            "wn:02084071n 1.0 2.0 3.0\n"
            "wn:00007846n 4.0 5.0 6.0\n"
            "madeupword0000 0.0 0.0 0.0\n"  # not a "wn:" token -- skipped
            "wn:99999999n 9.0 9.0 9.0\n"  # unresolvable offset -- skipped
        )
        vocabulary = SenseVocabulary.from_wn(
            ["omw-en-02084071-n", "omw-en-00007846-n"], nspecial=1
        )
        base = torch.full((len(vocabulary), 3), -1.0)

        result, matched = load_synset_centroid_vectors(
            vectors_file, vocabulary, base, progress=False
        )

        assert matched == 2
        assert result is base
        assert torch.equal(base[0], torch.tensor([-1.0, -1.0, -1.0]))  # reserved slot untouched
        low_index = vocabulary.index_for("omw-en-00007846-n")
        high_index = vocabulary.index_for("omw-en-02084071-n")
        assert torch.equal(base[low_index], torch.tensor([4.0, 5.0, 6.0]))
        assert torch.equal(base[high_index], torch.tensor([1.0, 2.0, 3.0]))

    def test_no_matches_leaves_base_untouched(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setitem(sys.modules, "wn", _fake_wn_module({}))
        vectors_file = tmp_path / "vectors.vec"
        vectors_file.write_text("wn:99999999n 1.0 2.0\n")
        vocabulary = SenseVocabulary.from_wn(["omw-en-02084071-n"], nspecial=0)
        base = torch.zeros(len(vocabulary), 2)

        _result, matched = load_synset_centroid_vectors(
            vectors_file, vocabulary, base, progress=False
        )

        assert matched == 0
        assert torch.equal(base, torch.zeros(len(vocabulary), 2))


class TestBuildSynsetCentroidVectorsFromLmms:
    def test_averages_sensekeys_sharing_a_synset_and_renormalizes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        bank_finance = _FakeSynset("omw-en-x-n")
        monkeypatch.setitem(
            sys.modules,
            "wn",
            _fake_wn_module(
                {},
                senses_by_lemma={
                    "bank": [
                        _FakeSense("bank%1:03:00::", bank_finance),
                        _FakeSense("bank%1:04:00::", bank_finance),
                    ]
                },
            ),
        )
        vectors_file = tmp_path / "lmms.vec"
        vectors_file.write_text("bank%1:03:00:: 1.0 0.0 0.0 0.0\nbank%1:04:00:: 0.0 1.0 0.0 0.0\n")
        vocabulary = SenseVocabulary.from_wn(["omw-en-x-n", "omw-en-y-n"], nspecial=0)
        base = torch.zeros(len(vocabulary), 4)

        _result, matched = build_synset_centroid_vectors_from_lmms(
            vectors_file, vocabulary, base, progress=False
        )

        assert matched == 1
        row = base[vocabulary.index_for("omw-en-x-n")]
        expected = torch.tensor([1.0, 1.0, 0.0, 0.0]) / (2**0.5)
        assert torch.allclose(row, expected, atol=1e-6)
        assert torch.isclose(row.norm(), torch.tensor(1.0), atol=1e-6)
        assert torch.equal(base[vocabulary.index_for("omw-en-y-n")], torch.zeros(4))

    def test_unresolvable_sensekeys_are_skipped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setitem(sys.modules, "wn", _fake_wn_module({}, senses_by_lemma={}))
        vectors_file = tmp_path / "lmms.vec"
        vectors_file.write_text("unknown%1:03:00:: 1.0 2.0\n")
        vocabulary = SenseVocabulary.from_wn(["omw-en-x-n"], nspecial=0)
        base = torch.zeros(len(vocabulary), 2)

        _result, matched = build_synset_centroid_vectors_from_lmms(
            vectors_file, vocabulary, base, progress=False
        )

        assert matched == 0

    def test_reduces_dimensionality_and_renormalizes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        synsets = {name: _FakeSynset(f"omw-en-{name}-n") for name in "abc"}
        monkeypatch.setitem(
            sys.modules,
            "wn",
            _fake_wn_module(
                {},
                senses_by_lemma={
                    name: [_FakeSense(f"{name}%1:01:00::", synset)]
                    for name, synset in synsets.items()
                },
            ),
        )
        vectors_file = tmp_path / "lmms.vec"
        vectors_file.write_text(
            "a%1:01:00:: 1.0 0.0 0.0 0.0\n"
            "b%1:01:00:: 0.0 1.0 0.0 0.0\n"
            "c%1:01:00:: 0.0 0.0 1.0 0.0\n"
        )
        vocabulary = SenseVocabulary.from_wn(
            ["omw-en-a-n", "omw-en-b-n", "omw-en-c-n"], nspecial=0
        )
        base = torch.zeros(len(vocabulary), 2)

        _result, matched = build_synset_centroid_vectors_from_lmms(
            vectors_file, vocabulary, base, target_dim=2, progress=False
        )

        assert matched == 3
        assert base.shape == (3, 2)
        for row in base:
            assert torch.isclose(row.norm(), torch.tensor(1.0), atol=1e-5)
