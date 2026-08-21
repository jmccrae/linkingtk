"""WordNet relation-graph construction for `StructuredLogits`.

Builds a sparse ``[V, V]`` adjacency over a
[SenseVocabulary][linkingtk.algorithms.wsd._ewiser_vocab.SenseVocabulary]
from `wn`'s own ``hypernym``/``similar``/``verb_group``/``derivation``
relations -- confirmed live against this repo's own installed lexicon that
all four are real, populated relation types (`derivation` is sense-level;
the other three are synset-level).

Reimplements the *algorithm* observed in EWISER's own graph builder
(`ResourceManager.make_adjacency_from_files`), not its code (that source is
CC-BY-NC-SA licensed, not redistributed here): for every node, gather every
incoming edge, normalize their weights to sum to 1 over *all* predecessors
first, **then** truncate to the top `max_incoming` by weight -- in that
order. The surviving top-k weights generally do **not** re-sum to 1, which
is a real, confirmed property of the reference's own behavior, not a bug
in this reimplementation.

Edge direction matches the reference: a hypernym relation edge is added
`(hyponym_index, hypernym_index)` -- the more general (and typically
better-attested) hypernym's logit propagates *down* into its more specific
hyponym, letting the model transfer confidence to senses it saw less
often in training. `similar`/`verb_group`/`derivation` are only added in
the direction each call reports -- confirmed directly that `wn` already
stores all three reciprocally (both synsets/senses on a pair list each
other), so the reverse direction is picked up for free when the loop
below reaches the other node; adding it a second time here would double
the edge weight.

Sourced natively via `wn` rather than EWISER's own BabelNet-id-keyed
`res/edges/*.tsv` files, avoiding both a heavy extra resource dependency
and the `bnids_map.txt` remapping step those files require. This means a
from-scratch graph cannot be verified bit-exactly against EWISER's own
(different relation coverage, different tie-breaking order) -- only
structurally, on synthetic data (see `tests/algorithms/wsd/test_ewiser_graph.py`).
Only relevant for
[EwiserTrainer][linkingtk.train.ewiser_trainer.EwiserTrainer]'s
from-scratch/continued-training path -- checkpoint-based inference uses
the checkpoint's own already-baked-in adjacency instead (see
[EwiserEncoder.from_checkpoint][linkingtk.algorithms.wsd.ewiser.EwiserEncoder.from_checkpoint]).
"""

from __future__ import annotations

import logging

import torch

from linkingtk.algorithms.wsd._ewiser_vocab import SenseVocabulary
from linkingtk.exceptions import OptionalDependencyError

logger = logging.getLogger("linkingtk")

_SYNSET_RELATIONS = ("hypernym", "similar", "verb_group")


def build_relation_adjacency(
    vocabulary: SenseVocabulary,
    lexicon: str = "omw-en:1.4",
    max_incoming: int = 5,
) -> torch.Tensor:
    """Build a sparse ``[len(vocabulary), len(vocabulary)]`` WordNet relation adjacency.

    Args:
        vocabulary: The sense inventory to build edges over -- only
            relation targets that are themselves in `vocabulary` become
            edges (a target outside `vocabulary` is silently dropped, the
            same "unresolved -> excluded, not an error" convention used
            elsewhere in this package, e.g.
            [UfsacDataset][linkingtk.datasets.ufsac.UfsacDataset]).
        lexicon: `wn` lexicon specifier to query relations against.
        max_incoming: Maximum incoming edges kept per node, by
            normalized weight (see module docstring for the
            normalize-then-truncate order).

    Returns:
        A coalesced sparse COO tensor, ``A[child, parent] = weight``.

    Raises:
        OptionalDependencyError: If `wn` isn't installed.
    """
    try:
        import wn
    except ImportError as exc:
        raise OptionalDependencyError("build_relation_adjacency", "wn") from exc

    incoming: dict[int, dict[int, float]] = {}

    def add_edge(child_index: int, parent_index: int) -> None:
        if child_index == parent_index:
            return
        node = incoming.setdefault(child_index, {})
        node[parent_index] = node.get(parent_index, 0.0) + 1.0

    for index in range(len(vocabulary)):
        synset_id = vocabulary.synset_id_for(index)
        if synset_id is None:
            continue
        try:
            synset = wn.synset(synset_id, lexicon=lexicon)
        except wn.Error:
            logger.warning(
                "build_relation_adjacency: %s not found in %s, skipping", synset_id, lexicon
            )
            continue

        relations = synset.relations(*_SYNSET_RELATIONS)
        for relation_type in _SYNSET_RELATIONS:
            for target in relations.get(relation_type, []):
                target_index = vocabulary.index_for(target.id)
                if target_index is not None:
                    add_edge(index, target_index)

        for sense in synset.senses():
            for target_sense in sense.relations("derivation").get("derivation", []):
                target_index = vocabulary.index_for(target_sense.synset().id)
                if target_index is not None:
                    add_edge(index, target_index)

    return _to_sparse_tensor(incoming, size=len(vocabulary), max_incoming=max_incoming)


def _to_sparse_tensor(
    incoming: dict[int, dict[int, float]], size: int, max_incoming: int
) -> torch.Tensor:
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for child_index, parents in incoming.items():
        total = sum(parents.values())
        normalized = sorted(
            ((parent_index, weight / total) for parent_index, weight in parents.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        for parent_index, weight in normalized[:max_incoming]:
            rows.append(child_index)
            cols.append(parent_index)
            values.append(weight)

    indices = torch.tensor([rows, cols], dtype=torch.long)
    values_tensor = torch.tensor(values, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values_tensor, size=(size, size)).coalesce()
