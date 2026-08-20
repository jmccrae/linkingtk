"""Benchmark-methodology test: GlossBertLinker trained on a tiny
UFSAC-shaped fixture corpus, through the real `UfsacDataset` -> `WnEntitySource`
-> `GlossBertLinker`/`CrossEncoderTrainer` pipeline.

Checks the *methodology* -- real sense-key-to-synset-id resolution
(`sensekey_to_synset_id`), `ExactMatch` querying a `WnEntitySource` (not a
materialized candidate list), `CrossEncoderTrainer`'s blocking-restricted
eval producing well-formed metrics -- not a literal comparison to
GlossBERT's published numbers. See `examples/glossbert_reproduction.py`
for that, against the paper's own published checkpoint and real UFSAC
data. `wn` is mocked entirely (same `sys.modules` injection as
`tests/sources/test_wn.py`/`tests/datasets/test_ufsac.py`), so this needs
no real lexicon download.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from linkingtk.algorithms.wsd.glossbert import GlossBertLinker  # noqa: E402
from linkingtk.blocking.exact import ExactMatch  # noqa: E402
from linkingtk.core.entity import Entity  # noqa: E402
from linkingtk.datasets.ufsac import UfsacDataset  # noqa: E402
from linkingtk.eval import Evaluator  # noqa: E402
from linkingtk.sources.wn import WnEntitySource  # noqa: E402
from linkingtk.train.arguments import TrainingArguments  # noqa: E402
from linkingtk.train.cross_encoder import CrossEncoderTrainer  # noqa: E402

_TINY_MODEL = "hf-internal-testing/tiny-random-BertForSequenceClassification"

_XML = """<?xml version="1.0" encoding="UTF-8"?>
<corpus>
  <document id="d001">
    <paragraph>
      <sentence id="s1">
        <word surface_form="He" lemma="he" pos="PRP"/>
        <word surface_form="caught" lemma="catch" pos="VBD"/>
        <word surface_form="a" lemma="a" pos="DT"/>
        <word surface_form="bass" lemma="bass" pos="NN" wn30_key="bass%1:05:00::"/>
        <word surface_form="fishing" lemma="fish" pos="VBG"/>
        <word surface_form="." lemma="." pos="."/>
      </sentence>
      <sentence id="s2">
        <word surface_form="A" lemma="a" pos="DT"/>
        <word surface_form="crane" lemma="crane" pos="NN" wn30_key="crane%1:06:00::"/>
        <word surface_form="lifted" lemma="lift" pos="VBD"/>
        <word surface_form="beams" lemma="beam" pos="NNS"/>
        <word surface_form="." lemma="." pos="."/>
      </sentence>
    </paragraph>
  </document>
</corpus>
"""


class _FakeWnError(Exception):
    pass


class _FakeSynset:
    def __init__(self, id: str, lemma: str, definition: str) -> None:
        self.id = id
        self._lemma = lemma
        self._definition = definition

    def words(self) -> list[_FakeWord]:
        return [_FakeWord(self._lemma)]

    def lemmas(self) -> list[str]:
        return [self._lemma]

    def definition(self) -> str:
        return self._definition


class _FakeWord:
    def __init__(self, lemma: str) -> None:
        self._lemma = lemma

    def lemma(self) -> str:
        return self._lemma

    def lexicon(self) -> _FakeLexicon:
        return _FakeLexicon()


class _FakeLexicon:
    language = "en"


class _FakeSense:
    def __init__(self, identifier: str, synset: _FakeSynset) -> None:
        self._identifier = identifier
        self._synset = synset

    def metadata(self) -> dict[str, str]:
        return {"identifier": self._identifier}

    def synset(self) -> _FakeSynset:
        return self._synset


_BASS_FISH = _FakeSynset("oewn-bass-fish", "bass", "a fish")
_BASS_MUSIC = _FakeSynset("oewn-bass-music", "bass", "low-frequency musical sound")
_CRANE_MACHINE = _FakeSynset("oewn-crane-machine", "crane", "a lifting machine")
_CRANE_BIRD = _FakeSynset("oewn-crane-bird", "crane", "a wading bird")

_SYNSETS_BY_LEMMA = {
    "bass": [_BASS_FISH, _BASS_MUSIC],
    "crane": [_CRANE_MACHINE, _CRANE_BIRD],
}
_SYNSETS_BY_ID = {s.id: s for synsets in _SYNSETS_BY_LEMMA.values() for s in synsets}
_SENSES_BY_LEMMA = {
    "bass": [_FakeSense("bass%1:05:00::", _BASS_FISH), _FakeSense("bass%1:07:00::", _BASS_MUSIC)],
    "crane": [
        _FakeSense("crane%1:06:00::", _CRANE_MACHINE),
        _FakeSense("crane%1:05:00::", _CRANE_BIRD),
    ],
}


@pytest.fixture(autouse=True)
def fake_wn(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = types.ModuleType("wn")

    def synsets(
        form: str | None = None,
        pos: str | None = None,
        *,
        lexicon: str | None = None,
        lang: str | None = None,
    ) -> list[_FakeSynset]:
        return _SYNSETS_BY_LEMMA.get(form or "", [])

    def synset(id: str, *, lexicon: str | None = None, lang: str | None = None) -> _FakeSynset:
        found = _SYNSETS_BY_ID.get(id)
        if found is None:
            raise module.Error(f"no such synset: {id}")  # type: ignore[attr-defined]
        return found

    def senses(
        form: str | None = None, pos: str | None = None, *, lexicon: str | None = None
    ) -> list[_FakeSense]:
        return _SENSES_BY_LEMMA.get(form or "", [])

    module.synsets = synsets  # type: ignore[attr-defined]
    module.synset = synset  # type: ignore[attr-defined]
    module.senses = senses  # type: ignore[attr-defined]
    module.Error = _FakeWnError  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wn", module)
    return module


@pytest.fixture
def fixture_source(tmp_path: Path) -> str:
    path = tmp_path / "corpus.xml"
    path.write_text(_XML)
    return str(path)


def test_ufsac_dataset_to_glossbert_linker_pipeline(tmp_path: Path, fixture_source: str) -> None:
    mentions, senses, ground_truth = UfsacDataset(source=fixture_source).load()
    assert isinstance(senses, WnEntitySource)
    assert len(mentions) == 2
    assert len(ground_truth) == 2

    linker = GlossBertLinker(model_name_or_path=_TINY_MODEL, max_length=32)
    mentions_by_id = {entity.id: entity for entity in mentions}
    pairs: list[tuple[Entity, Entity]] = []
    for mention_id, synset_id in ground_truth:
        sense_entity = senses.get(synset_id)
        assert sense_entity is not None
        pairs.append((mentions_by_id[mention_id], sense_entity))

    args = TrainingArguments(
        output_dir=str(tmp_path / "model"),
        learning_rate=1e-3,
        num_epochs=2,
        batch_size=4,
        negative_samples_ratio=2,
    )
    blocking = ExactMatch(top_k=10)
    CrossEncoderTrainer(model=linker.model, args=args, train_data=pairs, blocking=blocking).train()

    results = linker.link(mentions, senses, blocking=blocking)
    predictions = [(result.source_id, result.target_id) for result in results]
    report = Evaluator.evaluate(predictions=predictions, ground_truth=ground_truth)

    assert {source_id for source_id, _ in predictions} == {m.id for m in mentions}
    assert 0.0 <= report.metrics["precision@1"] <= 1.0
