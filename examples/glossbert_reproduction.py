"""Reproduces GlossBERT(Sent-CLS-WS)'s published results using the paper's
own checkpoint, through this repo's `GlossBertLinker`/`UfsacDataset` pipeline.

No training happens here -- this loads Huang et al.'s own published
checkpoint (https://arxiv.org/pdf/1908.07245.pdf, best-reported
configuration) directly into `GlossBertEncoder` (its `BertForSequenceClassification`
architecture is verbatim what the checkpoint contains) and evaluates it
via `GlossBertLinker.link()`, the exact same production code path a
freshly-trained model would use.

Setup:

1. Download the checkpoint from the link in
   https://github.com/HSLCY/GlossBERT#news (`Sent_CLS_WS.zip`) and place
   it at `~/Downloads/Sent_CLS_WS.zip` (or point `_CHECKPOINT_ZIP` below
   at wherever you put it). It's a plain
   `config.json`/`pytorch_model.bin`/`vocab.txt` triple, extracted once
   into `~/.cache/linkingtk/glossbert_checkpoint/` on first run.
2. Download UFSAC 2.1 (https://github.com/getalp/UFSAC#content-of-the-repository)
   and extract it to `~/data/ufsac-public-2.1/` (or point `_UFSAC_DIR`
   below elsewhere) -- this script reads its `raganato_*.xml` files, the
   same Raganato et al. (2017) WSD evaluation framework data GlossBERT's
   own paper reports numbers against.

Candidates are every WordNet sense of each mention's lemma
(`ExactMatch(top_k=50)` against a `WnEntitySource` -- wide enough that no
real word's sense count gets truncated), matching the paper's own
candidate generation (no gold-POS filtering, no popularity prior).

Published target (the paper's README, "this checkpoint" row -- the
checkpoint this script actually loads, not the original paper run):
SE07=72.1, SE2=77.7, SE3=75.9, SE13=76.8, SE15=79.3, ALL=77.2 (micro F1,
which reduces to precision@1 in this always-answered, one-prediction-
per-instance setting).

Run with: `uv run python examples/glossbert_reproduction.py`
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import torch

from linkingtk.algorithms.wsd.glossbert import GlossBertEncoder, GlossBertLinker
from linkingtk.blocking.exact import ExactMatch
from linkingtk.datasets.ufsac import UfsacDataset
from linkingtk.eval import Evaluator

_CHECKPOINT_ZIP = Path.home() / "Downloads" / "Sent_CLS_WS.zip"
_CHECKPOINT_DIR = Path.home() / ".cache" / "linkingtk" / "glossbert_checkpoint"
_UFSAC_DIR = Path.home() / "data" / "ufsac-public-2.1"

# (UFSAC filename stem, README column label, README's "this checkpoint" score)
_EVAL_SETS = [
    ("raganato_semeval2007", "SE07", 72.1),
    ("raganato_senseval2", "SE2", 77.7),
    ("raganato_senseval3", "SE3", 75.9),
    ("raganato_semeval2013", "SE13", 76.8),
    ("raganato_semeval2015", "SE15", 79.3),
    ("raganato_ALL", "ALL", 77.2),
]


def _ensure_checkpoint_extracted() -> Path:
    if not (_CHECKPOINT_DIR / "pytorch_model.bin").exists():
        _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(_CHECKPOINT_ZIP) as archive:
            archive.extractall(_CHECKPOINT_DIR)
    return _CHECKPOINT_DIR


def main() -> None:
    checkpoint_dir = _ensure_checkpoint_extracted()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = GlossBertEncoder.from_checkpoint(checkpoint_dir, max_length=512)
    linker = GlossBertLinker(model=encoder)
    linker.model.to(device)
    blocking = ExactMatch(top_k=50)

    print(f"{'dataset':<8} {'precision@1':>12} {'published':>10}")
    for stem, label, published in _EVAL_SETS:
        mentions, senses, ground_truth = UfsacDataset(source=str(_UFSAC_DIR / f"{stem}.xml")).load()

        results = linker.link(mentions, senses, blocking=blocking)
        predictions = [(result.source_id, result.target_id) for result in results]
        report = Evaluator.evaluate(predictions=predictions, ground_truth=ground_truth)
        score = report.metrics["precision@1"] * 100

        print(f"{label:<8} {score:>11.1f}% {published:>9.1f}%")


if __name__ == "__main__":
    main()
