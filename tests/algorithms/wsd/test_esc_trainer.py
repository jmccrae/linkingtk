"""Benchmark-methodology test: `EscTrainer` trained on a tiny UFSAC-shaped
fixture corpus, through the real `UfsacDataset` -> `WnEntitySource` ->
`EscEncoder`/`EscTrainer` pipeline.

Checks the *methodology* -- real sense-key-to-synset-id resolution,
`ExactMatch` querying a `WnEntitySource`, extractive-QA span-position
cross-entropy actually reducing loss over epochs -- not a literal
comparison to ESC's published numbers. See `examples/esc_reproduction.py`
for that, against the paper's own published checkpoint and real UFSAC
data. `wn` is mocked entirely (same convention as
`test_ewiser_trainer.py`), so this needs no real lexicon download.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from linkingtk.algorithms.wsd.esc import EscEncoder, EscLinker  # noqa: E402
from linkingtk.blocking.exact import ExactMatch  # noqa: E402
from linkingtk.datasets.ufsac import UfsacDataset  # noqa: E402
from linkingtk.eval import Evaluator  # noqa: E402
from linkingtk.sources.wn import WnEntitySource  # noqa: E402
from linkingtk.train.arguments import TrainingArguments  # noqa: E402
from linkingtk.train.esc_trainer import EscTrainer  # noqa: E402

_TINY_MODEL = "hf-internal-testing/tiny-random-BartModel"
_MAX_LENGTH = 256  # see test_esc.py's module docstring for why this needs to be generous

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


_BASS_FISH = _FakeSynset("omw-en-bass-fish", "bass", "a fish")
_BASS_MUSIC = _FakeSynset("omw-en-bass-music", "bass", "low-frequency musical sound")
_CRANE_MACHINE = _FakeSynset("omw-en-crane-machine", "crane", "a lifting machine")
_CRANE_BIRD = _FakeSynset("omw-en-crane-bird", "crane", "a wading bird")

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


def _precision_at_1(
    linker: EscLinker, mentions: list, senses: WnEntitySource, ground_truth: list
) -> float:
    results = linker.link(mentions, senses, blocking=ExactMatch())
    predictions = [(result.source_id, result.target_id) for result in results]
    return Evaluator.evaluate(predictions=predictions, ground_truth=ground_truth).metrics[
        "precision@1"
    ]


def test_ufsac_dataset_to_esc_trainer_pipeline(tmp_path: Path, fixture_source: str) -> None:
    mentions, senses, ground_truth = UfsacDataset(source=fixture_source).load()
    assert isinstance(senses, WnEntitySource)
    assert len(mentions) == 2
    assert len(ground_truth) == 2

    encoder = EscEncoder(model_name_or_path=_TINY_MODEL, max_length=_MAX_LENGTH)
    linker = EscLinker(encoder)

    args = TrainingArguments(
        output_dir=str(tmp_path / "model"),
        learning_rate=1e-4,
        num_epochs=15,
        batch_size=2,
    )
    trainer = EscTrainer(
        model=encoder,
        args=args,
        train_data=(mentions, ground_truth),
        senses=senses,
        eval_data=(mentions, ground_truth),
    )
    trainer.train()

    # loss_history, not a before/after precision@1 comparison, is the
    # robust signal here: with only 2 eval mentions x 2 candidates each,
    # the untrained model can already land at a ceiling precision@1 by
    # chance (50% per mention), leaving no room to visibly "improve" --
    # the QA span-position loss actually decreasing is what proves the
    # training path (not just the eval path) works.
    assert len(trainer.loss_history) == args.num_epochs
    assert trainer.loss_history[-1] < trainer.loss_history[0]

    assert len(trainer.eval_history) == args.num_epochs
    for epoch_report in trainer.eval_history:
        assert 0.0 <= epoch_report.metrics["Hits@1"] <= 1.0
    assert (tmp_path / "model" / "model.pt").exists()

    # Sanity check that scoring through the real EscLinker.link() path
    # (not just the trainer's own internal loss) still runs cleanly
    # post-training.
    after = _precision_at_1(linker, mentions, senses, ground_truth)
    assert 0.0 <= after <= 1.0


def test_gold_sense_not_among_candidates_is_dropped_not_a_crash(
    tmp_path: Path, fixture_source: str
) -> None:
    mentions, senses, ground_truth = UfsacDataset(source=fixture_source).load()
    encoder = EscEncoder(model_name_or_path=_TINY_MODEL, max_length=_MAX_LENGTH)

    # A ground-truth id that doesn't resolve to any candidate this mention's
    # blocking would ever surface -- must be skipped (logged), not raise.
    bogus_ground_truth = [(mentions[0].id, "omw-en-nonexistent-n"), *ground_truth[1:]]

    args = TrainingArguments(output_dir=str(tmp_path / "model"), num_epochs=1, batch_size=2)
    trainer = EscTrainer(
        model=encoder, args=args, train_data=(mentions, bogus_ground_truth), senses=senses
    )

    trainer.train()  # must not raise
