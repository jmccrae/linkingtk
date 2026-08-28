"""ChatEA linker for Entity Alignment.

Ports ChatEA (Jiang et al. 2024, "Unlocking the Power of Large Language
Models for Entity Alignment", https://aclanthology.org/2024.acl-long.408.pdf)
from its reference implementation at
https://github.com/DataArcTech/ChatEA -- an LLM re-ranker layered on top
of an embedding-based EA method's own top-K candidates, rather than a
standalone EA method. Filed as #22, a follow-up to #62
([SimpleHHEALinker][linkingtk.algorithms.ea.simple_hhea.SimpleHHEALinker]),
which is the reference's own base embedding method and this linker's
intended `base_linker`.

See [_chatea_context][linkingtk.algorithms.ea._chatea_context],
[_chatea_prompts][linkingtk.algorithms.ea._chatea_prompts], and
[_chatea_reasoning][linkingtk.algorithms.ea._chatea_reasoning] for the
individual ported pieces and their documented fidelity decisions vs. the
reference.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from linkingtk.algorithms.base import DEFAULT_BLOCKING, BaseLinker
from linkingtk.algorithms.ea._chatea_context import (
    EntityContext,
    NeighborIndex,
    generate_descriptions,
)
from linkingtk.algorithms.ea._chatea_prompts import build_system_prompt, resolve_dimensions
from linkingtk.algorithms.ea._chatea_reasoning import ReasoningConfig, rerank_candidates
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity, label_texts
from linkingtk.core.result import AlignmentResult
from linkingtk.core.source import EntitySource
from linkingtk.exceptions import LinkingTKError
from linkingtk.llm.client import LlmClient
from linkingtk.matchers import DEFAULT_MATCHER, Matcher
from linkingtk.utils.graph import Graph, to_triples

if TYPE_CHECKING:
    import numpy.typing as npt

TemporalTriple = tuple[str, str, str, "str | None", "str | None"]


class EmbeddingLinker(Protocol):
    """What `ChatEALinker` needs from its `base_linker` -- the same shape
    [linkingtk.eval.ranking][]'s own private `_ScoringLinker` Protocol
    uses, duplicated locally since that one is module-private. Every
    linker in `linkingtk.algorithms.ea` (e.g.
    [SimpleHHEALinker][linkingtk.algorithms.ea.simple_hhea.SimpleHHEALinker])
    already implements this."""

    def source_embedding(self, entity_id: str) -> npt.NDArray[np.floating[Any]]:
        """Vector used to score `entity_id` as a scored pair's source side."""

    def target_embedding(self, entity_id: str) -> npt.NDArray[np.floating[Any]]:
        """Vector used to score `entity_id` as a scored pair's target side."""


logger = logging.getLogger("linkingtk")


class ChatEALinker(BaseLinker):
    """Re-ranks an already-fitted embedding linker's top-K candidates with an LLM.

    Unlike most linkers in `linkingtk.algorithms.ea`, has no `fit()` --
    `base_linker` must already be fitted before construction (its
    embeddings are what generates candidates), and the LLM itself needs
    no training. This mirrors
    [LlmBaseLinker][linkingtk.algorithms.llm.LlmBaseLinker], which is
    also `fit()`-free for the same reason.

    Args:
        base_linker: An already-fitted linker exposing `source_embedding`/
            `target_embedding` (see
            [EmbeddingLinker][linkingtk.algorithms.ea.chatea.EmbeddingLinker]) --
            e.g. a fitted
            [SimpleHHEALinker][linkingtk.algorithms.ea.simple_hhea.SimpleHHEALinker].
        client: An [LlmClient][linkingtk.llm.client.LlmClient] (e.g. from
            [create_client][linkingtk.llm.client.create_client]). This
            class never imports `openai`/`anthropic` itself.
        top_k: Candidates considered per source entity, ranked by
            `base_linker`'s own cosine similarity. The reference's own
            default (20).
        neigh_num: Neighbor triples shown per entity in the reasoning
            prompt (the reference's own `--neigh 5` default for the
            main re-ranking pass).
        desc_neigh_num: Neighbor triples shown per entity when
            generating its one-sentence description (the reference's
            own `--neigh 25` default -- wider than `neigh_num` since
            description generation happens once per entity, not once
            per comparison).
        desc_max_tokens: Forwarded to the description-generation LLM
            call.
        threshold: If the base method's own top-2 candidate scores
            differ by more than this, accept the top-1 candidate
            immediately with no LLM call at all (the reference's own
            confidence shortcut).
        history_len: Chat turns (user+assistant pairs) kept in the
            running conversation history within one source entity's
            re-ranking.
        relation_names: `relation_id` -> display name, e.g. from
            [IcewsWikiDataset.load_relation_labels][linkingtk.datasets.icews.IcewsWikiDataset.load_relation_labels].
            Falls back to the raw id for any relation not present, or
            when omitted entirely.
        temporal_triples: Optional `(subject_id, relation_id, object_id,
            start_label, end_label)` temporal facts (e.g. from
            [IcewsWikiDataset.load_temporal_graphs][linkingtk.datasets.icews.IcewsWikiDataset.load_temporal_graphs]).
            When given, used instead of `graph` for neighbor rendering
            (strictly more informative -- same ids, plus time) and
            enables the time-scoring dimension. `None` (default)
            disables the time dimension entirely, matching
            `SimpleHHEALinker.fit`'s own convention for the same
            reference feature-toggle.
        use_name: Whether to score name similarity.
        use_desc: Whether to generate and score entity descriptions
            (requires `use_name`, see
            [resolve_dimensions][linkingtk.algorithms.ea._chatea_prompts.resolve_dimensions]).
        use_struct: Whether to score structural (neighbor-tuple)
            similarity.
        matching: Strategy used to resolve scored candidates into final
            links, same as every other linker's `matching` argument.

    Note:
        `blocking` is accepted for interface compliance but not used --
        candidates come from `base_linker`'s own embedding similarity,
        not a `BlockingStrategy`, same pattern
        [LlmBaseLinker][linkingtk.algorithms.llm.LlmBaseLinker]/
        [StringSimilarityLinker][linkingtk.algorithms.string_similarity.StringSimilarityLinker]
        already document for their own unused params. `dataset2` must be
        a `list[Entity]`, not an `EntitySource` -- candidate generation
        needs every target's embedding upfront.
    """

    def __init__(
        self,
        base_linker: EmbeddingLinker,
        client: LlmClient,
        top_k: int = 20,
        neigh_num: int = 5,
        desc_neigh_num: int = 25,
        desc_max_tokens: int = 80,
        threshold: float = 0.5,
        history_len: int = 3,
        relation_names: dict[str, str] | None = None,
        temporal_triples: list[TemporalTriple] | None = None,
        use_name: bool = True,
        use_desc: bool = True,
        use_struct: bool = True,
        matching: Matcher = DEFAULT_MATCHER,
    ) -> None:
        self.base_linker = base_linker
        self.client = client
        self.top_k = top_k
        self.neigh_num = neigh_num
        self.desc_neigh_num = desc_neigh_num
        self.desc_max_tokens = desc_max_tokens
        self.threshold = threshold
        self.history_len = history_len
        self.relation_names = relation_names or {}
        self.temporal_triples = temporal_triples
        self.use_name = use_name
        self.use_desc = use_desc
        self.use_struct = use_struct
        self.matching = matching

    def link(
        self,
        dataset1: list[Entity],
        dataset2: list[Entity] | EntitySource,
        graph: Graph = None,
        blocking: BlockingStrategy = DEFAULT_BLOCKING,
    ) -> list[AlignmentResult]:
        if isinstance(dataset2, EntitySource):
            raise LinkingTKError(
                "ChatEALinker.link() requires dataset2 as a list[Entity], not an "
                "EntitySource -- candidate generation needs every target's embedding upfront."
            )

        entity_names = {
            entity.id: label_texts(entity)[0]
            for entity in (*dataset1, *dataset2)
            if label_texts(entity)
        }
        triples = to_triples(graph) if graph is not None else []
        reasoning_index = NeighborIndex(
            triples, self.temporal_triples, entity_names, self.relation_names, self.neigh_num
        )
        desc_index = NeighborIndex(
            triples, self.temporal_triples, entity_names, self.relation_names, self.desc_neigh_num
        )

        candidates_by_source_id = self._top_k_candidates(dataset1, dataset2)

        use_time = self.temporal_triples is not None
        dims = resolve_dimensions(self.use_name, self.use_desc, self.use_struct, use_time)
        system_prompt = build_system_prompt(dims)

        descriptions: dict[str, str] = {}
        if self.use_desc:
            needed_ids = {
                entity_id
                for source_id, candidates in candidates_by_source_id.items()
                for entity_id in (source_id, *(cid for cid, _ in candidates))
            }
            descriptions = generate_descriptions(
                self.client, needed_ids, desc_index, max_tokens=self.desc_max_tokens
            )

        def context(entity_id: str) -> EntityContext:
            return EntityContext(
                entity_id=entity_id,
                name=entity_names.get(entity_id, entity_id),
                description=descriptions.get(entity_id, ""),
                neighbors=reasoning_index.neighbors(entity_id),
            )

        config = ReasoningConfig(dims=dims, threshold=self.threshold, history_len=self.history_len)

        candidates_by_source: dict[str, list[tuple[str, float]]] = {}
        for source_id, candidates in candidates_by_source_id.items():
            main_ctx = context(source_id)
            candidate_contexts = [(context(cid), score) for cid, score in candidates]
            scores = rerank_candidates(
                self.client, system_prompt, main_ctx, candidate_contexts, config
            )
            if scores:
                candidates_by_source[source_id] = list(scores.items())

        return self.matching.match(candidates_by_source)

    def _top_k_candidates(
        self, dataset1: list[Entity], dataset2: list[Entity]
    ) -> dict[str, list[tuple[str, float]]]:
        """Each source entity's top-`top_k` targets, ranked by
        `base_linker`'s own cosine similarity -- the reference's own
        candidate-generation step, done here instead of via a
        `BlockingStrategy` since it needs real similarity scores (for
        the confidence shortcut), not just a candidate set."""
        target_ids = [entity.id for entity in dataset2]
        target_matrix = np.stack([self.base_linker.target_embedding(t) for t in target_ids])
        result: dict[str, list[tuple[str, float]]] = {}
        for source in dataset1:
            source_vector = self.base_linker.source_embedding(source.id).reshape(1, -1)
            scores = cosine_similarity(source_vector, target_matrix)[0]
            ranked = sorted(zip(target_ids, scores, strict=True), key=lambda item: -item[1])
            result[source.id] = [
                (target_id, float(score)) for target_id, score in ranked[: self.top_k]
            ]
        return result
