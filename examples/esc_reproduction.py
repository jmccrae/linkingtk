"""Reproduces ESC's published results using the paper's own checkpoint,
through this repo's `EscLinker`/`UfsacDataset` pipeline.

No training happens here -- this loads Barba, Pasini & Navigli's own
released SemCor checkpoint (https://github.com/SapienzaNLP/esc, NAACL
2021) directly into `EscEncoder.from_checkpoint` and evaluates it via
`EscLinker.link()`, the exact same production code path a freshly trained
model would use. **This is a hard acceptance gate for issue #41, not a
best-effort verification pass**: measured numbers are expected to match
the published ones (within a stated, verified methodology difference, not
an assumed one).

Setup:

1. The paper's own SemCor checkpoint (the Google Drive link in the ESC
   README's "Checkpoints" section) at
   ``~/Downloads/escher_semcor_best.ckpt`` (or point `_CHECKPOINT_PATH`
   below elsewhere).
2. UFSAC 2.1 (https://github.com/getalp/UFSAC#content-of-the-repository)
   extracted to `_UFSAC_DIR` below -- same Raganato et al. (2017) WSD
   evaluation framework data ESC's own paper reports numbers against (same
   convention as ``ewiser_reproduction.py``).

Only SE07 and ALL are published in the reference's own README (``SE07:
76.3 | ALL: 80.7`` for the SemCor-only checkpoint) -- the other Raganato
splits are reported below for reference, without a stated target.

Candidates are restricted to each mention's own tagged part of speech,
matching the reference's own `WordNetDataset.init_dataset`
(``wn_offsets_from_lemmapos``, confirmed directly against
``esc/esc_dataset.py``) -- the same convention `ewiser_reproduction.py`
already uses for EWISER's own (separately confirmed) POS-restricted
candidate generation.

Run with: `uv run python examples/esc_reproduction.py`
"""

from __future__ import annotations

from pathlib import Path

import torch

from linkingtk.algorithms.wsd.esc import EscEncoder, EscLinker
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.blocking.exact import ExactMatch
from linkingtk.core.entity import Entity
from linkingtk.core.source import EntitySource
from linkingtk.datasets.ufsac import UfsacDataset
from linkingtk.eval import Evaluator

_CHECKPOINT_PATH = Path.home() / "Downloads" / "escher_semcor_best.ckpt"
_UFSAC_DIR = Path.home() / "data" / "ufsac-public-2.1"

# (UFSAC filename stem, README column label, published F1 or None)
_EVAL_SETS = [
    ("raganato_semeval2007", "SE07", 76.3),
    ("raganato_senseval2", "SE2", None),
    ("raganato_senseval3", "SE3", None),
    ("raganato_semeval2013", "SE13", None),
    ("raganato_semeval2015", "SE15", None),
    ("raganato_ALL", "ALL", 80.7),
]

_PENN_TO_WORDNET_POS = {"N": "n", "V": "v", "J": "a", "R": "r"}


def _wordnet_pos_of_synset_id(synset_id: str) -> str | None:
    pos = synset_id.rsplit("-", 1)[-1]
    if pos == "s":  # adjective satellite counts as a plain adjective for POS filtering
        return "a"
    return pos if pos in "nvar" else None


def _mention_wordnet_pos(mention: Entity) -> str | None:
    tag = mention.properties.get("pos")
    return _PENN_TO_WORDNET_POS.get(tag[0]) if tag else None


class _PosRestrictedMatch(BlockingStrategy):
    """Wraps `ExactMatch`, keeping only candidates whose WordNet POS matches
    the mention's own tagged POS (see module docstring)."""

    def __init__(self, top_k: int = 50) -> None:
        self._inner = ExactMatch(top_k=top_k)

    def candidate_pairs(
        self, dataset1: list[Entity], dataset2: list[Entity] | EntitySource
    ) -> list[tuple[Entity, Entity]]:
        pairs = self._inner.candidate_pairs(dataset1, dataset2)
        filtered = []
        for entity1, entity2 in pairs:
            mention_pos = _mention_wordnet_pos(entity1)
            if mention_pos is None or _wordnet_pos_of_synset_id(entity2.id) == mention_pos:
                filtered.append((entity1, entity2))
        return filtered


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    blocking = _PosRestrictedMatch(top_k=50)

    encoder = EscEncoder.from_checkpoint(_CHECKPOINT_PATH, forward_batch_size=8)
    encoder.to(device)
    linker = EscLinker(encoder)

    print(f"{'dataset':<8} {'precision@1':>12} {'published':>10}")
    for stem, eval_label, published in _EVAL_SETS:
        mentions, senses, ground_truth = UfsacDataset(source=str(_UFSAC_DIR / f"{stem}.xml")).load()

        results = linker.link(mentions, senses, blocking=blocking)
        predictions = [(result.source_id, result.target_id) for result in results]
        report = Evaluator.evaluate(predictions=predictions, ground_truth=ground_truth)
        score = report.metrics["precision@1"] * 100

        published_str = f"{published:.1f}%" if published is not None else "n/a"
        print(f"{eval_label:<8} {score:>11.1f}% {published_str:>10}")


if __name__ == "__main__":
    main()
