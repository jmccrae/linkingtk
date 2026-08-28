"""RDGCN knowledge-graph-embedding linker for Entity Alignment.

Wu, Y., Liu, X., Feng, Y., Wang, Z., Yan, R., & Zhao, D. (2019).
Relation-Aware Entity Alignment for Heterogeneous Knowledge Graphs. IJCAI
2019. https://www.ijcai.org/proceedings/2019/0733.pdf

The second GNN-based EA linker in this package (see #18's parent issue and
#42's [GCNAlignLinker][linkingtk.algorithms.ea.gcn_align.GCNAlignLinker]).
RDGCN builds entity representations from **two graphs jointly**: the
"primal" graph over entities (structural, like GCN-Align's), and a "dual"
graph over *relations* (two relations are dual-graph-adjacent if their
head-/tail-entity sets overlap a lot), attending back and forth between
them for two rounds before a final 2-layer diagonal-weight GCN with highway
gates. This is a faithful port of OpenEA's reference implementation
(https://github.com/nju-websoft/OpenEA/blob/master/src/openea/approaches/rdgcn.py),
not a from-scratch reading of the paper -- see
[_rdgcn_training][linkingtk.algorithms.ea._rdgcn_training] and
[_rdgcn_torch][linkingtk.algorithms.ea._rdgcn_torch] for the algorithm's
own docstrings, and this module's deviations below (confirmed against
OpenEA's own reference config, ``run/args/rdgcn_args_15K.json``, and
source):

- **Requires [EnFr15KAttrDataset][linkingtk.datasets.EnFr15KAttrDataset],
  not [EnFr15KDataset][linkingtk.datasets.EnFr15KDataset].** RDGCN's entity
  representations are seeded from pretrained word-vector embeddings of
  each entity's *local name* -- ``EnFr15KDataset``'s numeric-id,
  URI-free format has no name text to embed (same reason
  [KDCoELinker][linkingtk.algorithms.ea.kdcoe.KDCoELinker] (#29) needs it).
  ``fit()`` raises `LinkingTKError` early if no entity in ``dataset1``/
  ``dataset2`` has a non-empty label, as a cheap guard.
- **Local names come from `Entity.labels`, not OpenEA's own URI-parsing.**
  OpenEA's ``_get_local_name_by_name_triple`` has a dataset-specific
  name-attribute matching step (special-cased predicate lists for ``D_Y``/
  ``D_W``) that falls through to a generic ``else: name_attribute_list = {}``
  for every other dataset, including EN-FR-15K -- meaning OpenEA's own code
  already just uses the entity URI's tail segment for this dataset, the
  same text this repo's loader already populates as the label. Confirmed a
  no-op deviation for this benchmark, not a fidelity-losing simplification.
- **Pretrained word vectors via `_kdcoe_text.load_fasttext_vectors`**,
  reusing exactly the file OpenEA's own reference hardcodes
  (``wiki-news-300d-1M.vec.zip``) and the same streaming-fetch mechanism
  [KDCoELinker][linkingtk.algorithms.ea.kdcoe.KDCoELinker] already
  established, rather than a separate implementation.
- **Hard-negative mining excludes each anchor's own id from its own
  candidate pool** -- OpenEA's own ``get_neg`` doesn't, which (since an
  entity's distance to itself is always the global minimum) makes an
  anchor frequently rank as its own "hardest negative," a training signal
  that's structurally meaningless. See
  [mine_hard_negatives][linkingtk.algorithms.ea._rdgcn_torch.mine_hard_negatives]'s
  docstring.
- **Duplicate-relation primal edges are coalesce-summed, not kept as
  separate softmax competitors** -- a `torch.sparse.softmax`-driven
  simplification vs. TensorFlow's uncoalesced-sparse-tensor behavior; see
  [_SparseAttentionLayer][linkingtk.algorithms.ea._rdgcn_torch.build_rdgcn_model]'s
  docstring (inside ``build_rdgcn_model``).
- **Candidate scoring uses Manhattan (L1) distance, not cosine
  similarity.** OpenEA's own config for this method is ``eval_metric:
  "manhattan"``, ``eval_norm: false`` (confirmed by reading
  ``run/args/rdgcn_args_15K.json`` directly) -- and
  [margin_ranking_loss_l1][linkingtk.algorithms.ea._ea_losses.margin_ranking_loss_l1]
  itself trains embeddings to be close in L1 distance, not high-cosine.
  This is a real correctness fix matching the training geometry, not
  just fidelity to OpenEA's own choice -- confirmed empirically: on the
  real EN-FR-15K-V1 benchmark, switching the *evaluation* metric alone
  (same trained embeddings, via
  [rank_exhaustive][linkingtk.eval.ranking.rank_exhaustive]'s ``metric``/
  ``csls_k``) moved Hits@1 from 0.666 (cosine) to 0.735
  (manhattan+CSLS k=10, OpenEA's own config) against a published 0.755 --
  closing the gap from ~88% to ~97% relative. CSLS (Cross-domain
  Similarity Local Scaling) is a further, separate refinement used by
  ``examples/rdgcn_benchmark.py`` but not by ``link()`` itself (CSLS
  needs the full candidate pool's neighbor structure, which doesn't fit
  ``link()``'s post-blocking, per-source candidate-list shape).
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial.distance import cdist

from linkingtk.algorithms.base import DEFAULT_BLOCKING, BaseLinker
from linkingtk.algorithms.ea._ea_losses import margin_ranking_loss_l1
from linkingtk.algorithms.ea._kdcoe_text import load_fasttext_vectors, tokenize_description
from linkingtk.algorithms.ea._rdgcn_torch import build_rdgcn_model, mine_hard_negatives
from linkingtk.algorithms.ea._rdgcn_training import (
    build_dual_graph,
    build_edge_relations,
    build_primal_adjacency,
    build_relation_masks,
    init_name_embeddings,
)
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity, label_texts
from linkingtk.core.result import AlignmentResult
from linkingtk.core.source import EntitySource
from linkingtk.exceptions import LinkingTKError, OptionalDependencyError
from linkingtk.matchers import DEFAULT_MATCHER, Matcher
from linkingtk.utils.device import resolve_device
from linkingtk.utils.graph import Graph, build_id_mappings, map_triples_to_ids, to_triples
from linkingtk.utils.sparse_gcn import coo_to_torch_sparse, normalize_adjacency_coo

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

_FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"
_NEG_REFRESH_INTERVAL = 10


class RDGCNLinker(BaseLinker):
    """Scores candidate pairs via RDGCN's dual/primal-attention-plus-GCN embeddings.

    Must be [fit][linkingtk.algorithms.ea.rdgcn.RDGCNLinker.fit] before
    [link][linkingtk.algorithms.base.BaseLinker.link] can be called. See
    the module docstring for what's ported vs. deliberately deviated from
    OpenEA, and why `dataset1`/`dataset2` must come from
    [EnFr15KAttrDataset][linkingtk.datasets.EnFr15KAttrDataset] rather than
    [EnFr15KDataset][linkingtk.datasets.EnFr15KDataset].

    Args:
        embedding_dim: Entity-embedding dimensionality -- must equal the
            pretrained word vectors' own dimensionality (no projection
            layer sits between them). OpenEA's published EN-FR-15K-V1
            config uses ``300`` (fastText's native size).
        num_epochs: Training epochs. OpenEA's config allows up to ``2000``
            with early stopping.
        learning_rate: Adam's learning rate. OpenEA's published value is
            ``0.002``.
        neg_triple_num: Hard negatives mined per seed pair per side (``k``),
            refreshed every 10 epochs. OpenEA's published value is ``125``.
        gamma: Margin for
            [margin_ranking_loss_l1][linkingtk.algorithms.ea._ea_losses.margin_ranking_loss_l1].
            OpenEA's published value is ``1.0``.
        alpha: Round-1 primal-embedding mixing weight. OpenEA's published
            value is ``0.1``.
        beta: Round-2 primal-embedding mixing weight. OpenEA's published
            value is ``0.3``.
        default_name_length: Max local-name tokens summed per entity for
            the initial embedding. OpenEA's published value is ``4``.
        word_embed_url: URL of a fastText-format ``.vec.zip`` (see
            ``_kdcoe_text.py``'s ``load_fasttext_vectors``). Defaults to
            the real, published pretrained file; tests override this with
            a ``file://`` url pointing at a tiny local fixture.
        cache_dir: Passed through to ``fetch_cached`` for the word-vector
            download. ``None`` uses ``fetch_cached``'s own default cache
            location.
        matching: Strategy used to resolve scored candidates into final
            links. Defaults to
            [GreedyMatcher][linkingtk.matchers.greedy.GreedyMatcher].
        device: Torch device to train on, e.g. ``"cpu"`` (default) or
            ``"cuda"``/``"cuda:0"``. Trained embeddings are always stored
            as CPU numpy arrays regardless of this setting.
    """

    def __init__(
        self,
        embedding_dim: int = 300,
        num_epochs: int = 500,
        learning_rate: float = 0.002,
        neg_triple_num: int = 125,
        gamma: float = 1.0,
        alpha: float = 0.1,
        beta: float = 0.3,
        default_name_length: int = 4,
        word_embed_url: str = _FASTTEXT_URL,
        cache_dir: Path | None = None,
        matching: Matcher = DEFAULT_MATCHER,
        device: str = "cpu",
    ) -> None:
        self.embedding_dim = embedding_dim
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.neg_triple_num = neg_triple_num
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.default_name_length = default_name_length
        self.word_embed_url = word_embed_url
        self.cache_dir = cache_dir
        self.matching = matching
        self.device = device
        self._id_to_vector: dict[str, npt.NDArray[np.floating[Any]]] = {}
        self._fitted = False

    def fit(
        self,
        dataset1: list[Entity],
        dataset2: list[Entity],
        ground_truth: list[tuple[str, str]],
        graph: Graph,
        random_state: int | None = None,
        val_ground_truth: list[tuple[str, str]] | None = None,
        patience: int = 5,
        eval_every: int = 10,
    ) -> RDGCNLinker:
        """Train the two-round dual/primal-attention GCN over the combined graph.

        Args:
            dataset1: Source entities -- must carry non-empty `labels`
                (e.g. from
                [EnFr15KAttrDataset][linkingtk.datasets.EnFr15KAttrDataset]),
                used to seed each entity's initial embedding from
                pretrained word vectors of its local name.
            dataset2: Target entities. See ``dataset1``.
            ground_truth: List of ``(source_id, target_id)`` known-correct
                pairs used as positive pairs in the margin loss.
            graph: The combined relational structure of both KGs (e.g.
                ``to_triples(graph1) + to_triples(graph2)`` from a
                [GraphDatasetLoader][linkingtk.datasets.base.GraphDatasetLoader]'s
                ``load_graphs()``) -- entity ids on both sides must already
                be disjoint, as they are from that loader.
            random_state: Seed for reproducible training. Left unspecified,
                training is non-deterministic.
            val_ground_truth: Optional held-out pairs used for early
                stopping -- every ``eval_every`` epochs, Hits@1 is checked
                against this set, and training stops after ``patience``
                checks with no improvement. If ``None`` (default), trains
                the full ``num_epochs`` unconditionally.
            patience: Number of non-improving ``eval_every``-spaced checks
                to tolerate before stopping early. Only used if
                ``val_ground_truth`` is given.
            eval_every: How often (in epochs) to check ``val_ground_truth``.
                Only used if ``val_ground_truth`` is given.

        Returns:
            ``self``, for chaining.

        Raises:
            LinkingTKError: If no entity in ``dataset1``/``dataset2`` has a
                non-empty label, if none of ``ground_truth``'s pairs have
                both ids present in ``graph``'s own triples, or if
                ``device`` is invalid or unavailable.
            OptionalDependencyError: If torch isn't installed.
        """
        entities = dataset1 + dataset2
        if not any(label_texts(entity) for entity in entities):
            raise LinkingTKError(
                "RDGCNLinker.fit() needs entities with non-empty `labels` to seed name "
                "embeddings from -- pass entities from EnFr15KAttrDataset (or similar), "
                "not EnFr15KDataset. See the module docstring."
            )

        try:
            import torch
        except ImportError as exc:
            raise OptionalDependencyError("RDGCNLinker", "kge") from exc

        device = resolve_device(self.device)
        if random_state is not None:
            torch.manual_seed(random_state)
            torch.cuda.manual_seed_all(random_state)

        triples = to_triples(graph)
        entity_to_id, relation_to_id = build_id_mappings(triples)
        mapped = map_triples_to_ids(triples, entity_to_id, relation_to_id)
        num_entities = len(entity_to_id)
        num_relations = len(relation_to_id)

        seed_pairs = [
            (entity_to_id[s], entity_to_id[t])
            for s, t in ground_truth
            if s in entity_to_id and t in entity_to_id
        ]
        if not seed_pairs:
            raise LinkingTKError(
                "None of `ground_truth`'s pairs have both ids present in `graph`'s "
                "own triples; fit() has no seed pairs to train RDGCN's margin loss with."
            )

        entity_by_id = {entity.id: entity for entity in entities}
        vocabulary = {
            token
            for entity in entities
            for token in tokenize_description(label_texts(entity)[0] if label_texts(entity) else "")
        }
        word_vectors = load_fasttext_vectors(self.word_embed_url, vocabulary, self.cache_dir)
        ordered_entities = [
            entity_by_id.get(entity_id, Entity(id=entity_id, labels=[]))
            for entity_id, _index in sorted(entity_to_id.items(), key=lambda item: item[1])
        ]
        name_embeddings = init_name_embeddings(
            ordered_entities, word_vectors, self.embedding_dim, self.default_name_length
        )
        initial_embeds_np = np.stack(
            [name_embeddings[entity.id] for entity in ordered_entities]
        ).astype(np.float32)
        initial_embeds = torch.from_numpy(initial_embeds_np).to(device)

        primal_indices, primal_values = build_primal_adjacency(mapped, num_entities)
        primal_indices, primal_values = normalize_adjacency_coo(
            primal_indices, primal_values, num_entities
        )
        primal_adjacency = coo_to_torch_sparse(
            primal_indices, primal_values, (num_entities, num_entities), device
        )

        dual_adjacency_np = build_dual_graph(mapped, num_relations, num_entities)
        dual_adjacency = torch.from_numpy(dual_adjacency_np).float().to(device)

        (head_indices, head_values), (tail_indices, tail_values) = build_relation_masks(
            mapped, num_relations
        )
        head_mask = coo_to_torch_sparse(
            head_indices, head_values, (num_relations, num_entities), device
        )
        tail_mask = coo_to_torch_sparse(
            tail_indices, tail_values, (num_relations, num_entities), device
        )

        edge_heads_np, edge_tails_np, edge_relations_np = build_edge_relations(mapped)
        edge_heads = torch.from_numpy(edge_heads_np).long().to(device)
        edge_tails = torch.from_numpy(edge_tails_np).long().to(device)
        edge_relations = torch.from_numpy(edge_relations_np).long().to(device)

        model = build_rdgcn_model(initial_embeds, self.embedding_dim, self.alpha, self.beta).to(
            device
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        pos_left = torch.tensor([s for s, _ in seed_pairs], device=device)
        pos_right = torch.tensor([t for _, t in seed_pairs], device=device)

        def _forward() -> torch.Tensor:
            return model(  # type: ignore[no-any-return]
                primal_adjacency,
                dual_adjacency,
                head_mask,
                tail_mask,
                edge_heads,
                edge_tails,
                edge_relations,
            )

        with torch.no_grad():
            neg_left, neg_right, neg2_left, neg2_right = mine_hard_negatives(
                _forward(), pos_left, pos_right, self.neg_triple_num
            )

        val_pairs = [
            (s, t) for s, t in (val_ground_truth or []) if s in entity_to_id and t in entity_to_id
        ]
        best_hits1 = -1.0
        epochs_without_improvement = 0

        for epoch in range(self.num_epochs):
            if epoch % _NEG_REFRESH_INTERVAL == 0 and epoch > 0:
                with torch.no_grad():
                    neg_left, neg_right, neg2_left, neg2_right = mine_hard_negatives(
                        _forward(), pos_left, pos_right, self.neg_triple_num
                    )

            embeddings = _forward()
            loss = margin_ranking_loss_l1(
                embeddings,
                pos_left,
                pos_right,
                neg_left,
                neg_right,
                neg2_left,
                neg2_right,
                self.gamma,
            )
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

            if val_pairs and (epoch + 1) % eval_every == 0:
                with torch.no_grad():
                    current_embeds = _forward().cpu().numpy()
                hits1 = _validation_hits1(current_embeds, entity_to_id, val_pairs)
                if hits1 <= best_hits1:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        break
                else:
                    best_hits1 = hits1
                    epochs_without_improvement = 0

        with torch.no_grad():
            final_embeds = _forward().cpu().numpy()
        self._id_to_vector = {
            entity_id: final_embeds[index] for entity_id, index in entity_to_id.items()
        }
        self._fitted = True
        return self

    def link(
        self,
        dataset1: list[Entity],
        dataset2: list[Entity] | EntitySource,
        graph: Graph = None,
        blocking: BlockingStrategy = DEFAULT_BLOCKING,
    ) -> list[AlignmentResult]:
        if not self._fitted:
            raise LinkingTKError("RDGCNLinker.link() called before fit().")

        pairs = blocking.candidate_pairs(dataset1, dataset2)
        target_ids_by_source: dict[str, list[str]] = defaultdict(list)
        for entity1, entity2 in pairs:
            target_ids_by_source[entity1.id].append(entity2.id)

        candidates_by_source: dict[str, list[tuple[str, float]]] = {}
        for source_id, target_ids in target_ids_by_source.items():
            source_vector = self.source_embedding(source_id).reshape(1, -1)
            target_matrix = np.stack([self.target_embedding(target_id) for target_id in target_ids])
            scores = _manhattan_similarity(source_vector, target_matrix)[0]
            candidates_by_source[source_id] = list(
                zip(target_ids, (float(score) for score in scores), strict=True)
            )

        return self.matching.match(candidates_by_source)

    def source_embedding(self, entity_id: str) -> npt.NDArray[np.floating[Any]]:
        """Vector used to score ``entity_id`` as a scored pair's source side."""
        return self._embedding(entity_id)

    def target_embedding(self, entity_id: str) -> npt.NDArray[np.floating[Any]]:
        """Vector used to score ``entity_id`` as a scored pair's target side."""
        return self._embedding(entity_id)

    def _embedding(self, entity_id: str) -> npt.NDArray[np.floating[Any]]:
        vector = self._id_to_vector.get(entity_id)
        if vector is None:
            raise LinkingTKError(
                f"Entity {entity_id!r} has no trained embedding -- it didn't appear "
                "in fit()'s `graph`."
            )
        return vector


def _validation_hits1(
    embeds: npt.NDArray[np.floating[Any]],
    entity_to_id: dict[str, int],
    val_pairs: list[tuple[str, str]],
) -> float:
    """Cheap top-1 match rate over ``val_pairs``, for early-stopping only.

    Same deliberately-simplified stand-in as
    [GCNAlignLinker][linkingtk.algorithms.ea.gcn_align.GCNAlignLinker]'s
    ``_validation_hits1``.
    """
    sources = [s for s, _ in val_pairs]
    targets = [t for _, t in val_pairs]
    source_matrix = np.stack([embeds[entity_to_id[s]] for s in sources])
    target_matrix = np.stack([embeds[entity_to_id[t]] for t in targets])
    similarities = _manhattan_similarity(source_matrix, target_matrix)
    predicted = np.argmax(similarities, axis=1)
    correct = sum(1 for i, j in enumerate(predicted) if j == i)
    return correct / len(val_pairs)


def _manhattan_similarity(
    source_matrix: npt.NDArray[np.floating[Any]], target_matrix: npt.NDArray[np.floating[Any]]
) -> npt.NDArray[np.floating[Any]]:
    """``1 - L1 distance`` -- matches
    [margin_ranking_loss_l1][linkingtk.algorithms.ea._ea_losses.margin_ranking_loss_l1]'s
    training geometry, unlike cosine similarity. See the module docstring.
    """
    result: npt.NDArray[np.floating[Any]] = 1 - cdist(
        source_matrix, target_matrix, metric="cityblock"
    )
    return result
