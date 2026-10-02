"""Part-of-speech filtering for WordNet sense candidates."""

from __future__ import annotations

from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity
from linkingtk.core.source import EntitySource

# Penn Treebank tag prefix -> WordNet synset-id POS suffixes (adjectives
# include satellites, "-s").
_PENN_TO_WN = {"NN": ("n",), "VB": ("v",), "JJ": ("a", "s"), "RB": ("r",)}


class WordNetPosFilter(BlockingStrategy):
    """Drops WordNet candidates whose part of speech doesn't match the mention's.

    [WnEntitySource.search][linkingtk.sources.wn.WnEntitySource.search]
    looks a lemma up across every POS, so a verb mention like "play" is
    offered noun senses first. That makes "first candidate" a poor
    most-frequent-sense baseline (0.305 instead of 0.536 on SemEval-2007),
    and gives a linker extra wrong-POS candidates to choose from.

    The mention's Penn Treebank tag is read from `properties["pos"]` (as
    [UfsacDataset][linkingtk.datasets.ufsac.UfsacDataset] provides it) and
    compared with the POS suffix of each candidate's synset id (e.g.
    `omw-en-01024190-v`). Mentions without a recognised tag keep all of
    their candidates. Candidate order is preserved.

    Args:
        inner: The blocking strategy whose candidates are filtered, e.g.
            `ExactMatch(top_k=50)` against a `WnEntitySource`.
    """

    def __init__(self, inner: BlockingStrategy) -> None:
        self.inner = inner

    def candidate_pairs(
        self, dataset1: list[Entity], dataset2: list[Entity] | EntitySource
    ) -> list[tuple[Entity, Entity]]:
        return [
            (mention, sense)
            for mention, sense in self.inner.candidate_pairs(dataset1, dataset2)
            if _same_pos(mention, sense)
        ]


def _same_pos(mention: Entity, sense: Entity) -> bool:
    suffixes = _PENN_TO_WN.get(mention.properties.get("pos", "")[:2])
    return suffixes is None or sense.id.rsplit("-", 1)[-1] in suffixes
