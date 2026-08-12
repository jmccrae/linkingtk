"""JAPE knowledge-graph-embedding linker for Entity Alignment.

Sun, Z., Hu, W., & Li, C. (2017). Cross-Lingual Entity Alignment via Joint
Attribute-Preserving Embedding. ISWC 2017.
https://link.springer.com/chapter/10.1007/978-3-319-68288-4_37

Like [MTransELinker][linkingtk.algorithms.ea.mtranse.MTransELinker] and
[IPTransELinker][linkingtk.algorithms.ea.iptranse.IPTransELinker], this is a
faithful reimplementation of JAPE's actual training procedure, ported
directly from OpenEA's reference implementation
(https://github.com/nju-websoft/OpenEA/blob/master/src/openea/approaches/jape.py
and .../attr2vec.py) rather than from a from-scratch reading of the paper.
JAPE combines two signals:

- **Structural embedding**: the same shared-id ("sharing") mechanism as
  `IPTransELinker` -- reuses ``_iptranse_training.py``'s
  ``build_shared_id_mappings`` and ``sample_negative_triples`` directly --
  trained via ``sum(||h+r-t||^2) - neg_alpha * sum(||h'+r-t'||^2)``.
  Unlike MTransE/IPTransE, this has **no margin/hinge** -- it's OpenEA's
  actual (unbounded-below) formula for JAPE specifically, not a
  simplification.
- **Attribute-correlation embedding**: a skip-gram-style model (ported from
  OpenEA's ``Attr2Vec``) trained over which attribute predicates co-occur
  on the same entity -- entities that are a known seed pair have their
  attribute vocabularies merged before training, which is what makes this
  cross-lingual-aware. The trained attribute embeddings produce a
  per-entity attribute vector (mean-pooled, L2-normalized); a similarity
  matrix between the two KGs' not-yet-seeded entities, thresholded to keep
  only confident correlations, then **regularizes the structural
  embeddings during joint training** (not a post-hoc concatenation): each
  epoch, a random subsample of KG1's reference entities has its structural
  embedding pulled toward an attribute-similarity-weighted blend of KG2's
  structural embeddings, via a separate optimizer.

Deviations from OpenEA's own code:

- **`fit()` gains two new optional parameters**, `attribute_triples1`/
  `attribute_triples2` -- the paper's method genuinely doesn't fit the EA
  linker family's existing signature otherwise, since attribute triples
  are JAPE's whole point. Left as `None` (default), `fit()` degrades to
  structural-only training (a legitimate, still-functional mode -- just
  not what's benchmarked against JAPE's published number, which needs the
  attribute signal). Most of this repo's EA dataset loaders have no
  attribute triples at all; see
  [linkingtk.datasets.openea_native][]'s module docstring for the one
  that does and why it's a separate loader family from
  [linkingtk.datasets.openea][]'s.
- **Attr2Vec's NCE loss uses uniform negative sampling**, not OpenEA's
  `tf.nn.nce_loss` default (a log-uniform sampler assuming frequency-
  sorted class ids, which our attribute-predicate ids aren't, and which
  PyTorch has no drop-in equivalent for) -- see ``_jape_torch.py``'s
  ``train_attr2vec`` docstring.
- **The attribute-similarity reference pool** is "every `dataset1`/
  `dataset2` entity not already a `ground_truth` source/target" rather
  than OpenEA's own `valid_entities + test_entities` -- the same
  deviation, for the same reason, already documented on
  [IPTransELinker][linkingtk.algorithms.ea.iptranse.IPTransELinker]'s
  bootstrap pool.
- **The attribute-similarity matrix is sparsified to each row's single
  best match**, not left as OpenEA's dense, thresholded-only matrix --
  see ``_jape_training.py``'s ``sparsify_to_row_argmax`` docstring for
  why (found while benchmarking on real EN-FR-15K-V1 data: attribute-
  predicate mean-pooled vectors are coarse enough at this scale that the
  dense blend measurably hurt Hits@1 relative to structural-only
  training, and reducing ``attr_sim_mat_beta`` didn't fix it).
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from linkingtk.algorithms.base import DEFAULT_BLOCKING, BaseLinker
from linkingtk.algorithms.ea._iptranse_training import build_shared_id_mappings
from linkingtk.algorithms.ea._jape_torch import (
    build_kg_context,
    train_attr2vec,
    train_sim_epoch,
    train_triple_epoch,
    validation_hits1,
)
from linkingtk.algorithms.ea._jape_training import (
    build_entity_attributes,
    generate_training_pairs,
    pool_entity_attribute_vectors,
    reference_pools,
    select_popular_attributes,
    sparsify_to_row_argmax,
)
from linkingtk.algorithms.matching import DEFAULT_MATCHER, Matcher
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity
from linkingtk.core.result import AlignmentResult
from linkingtk.exceptions import LinkingTKError, OptionalDependencyError
from linkingtk.utils.graph import Graph, Triple, map_triples_to_ids, to_triples

if TYPE_CHECKING:
    import numpy.typing as npt


class JAPELinker(BaseLinker):
    """Scores candidate pairs via a shared TransE space regularized by attribute similarity.

    Must be [fit][linkingtk.algorithms.ea.jape.JAPELinker.fit] before
    [link][linkingtk.algorithms.base.BaseLinker.link] can be called. See
    the module docstring for how this differs from
    [MTransELinker][linkingtk.algorithms.ea.mtranse.MTransELinker]/
    [IPTransELinker][linkingtk.algorithms.ea.iptranse.IPTransELinker].

    Args:
        embedding_dim: Dimensionality of the trained structural and
            attribute embeddings. OpenEA's published EN-FR-15K-V1 config
            uses ``100``.
        num_epochs: Structural training epochs. OpenEA's config allows up
            to ``2000`` with early stopping (see ``val_ground_truth`` on
            [fit][linkingtk.algorithms.ea.jape.JAPELinker.fit]).
        batch_size: Mini-batch size for both the structural triple loss
            and (if attribute triples are given) Attr2Vec's training.
        learning_rate: Adagrad's learning rate, for every optimizer used
            here -- OpenEA's published value is ``0.01``.
        neg_alpha: Weight on the negative-triple term in the structural
            loss (``pos_loss - neg_alpha * neg_loss``). OpenEA's published
            value is ``0.1``.
        top_attr_threshold: Fraction of each KG's distinct attribute
            predicates to keep, by descending frequency. OpenEA's
            published value is ``0.9``.
        attr_sim_mat_threshold: Attribute-similarity values below this are
            zeroed out before being used as a structural regularizer.
            OpenEA's published value is ``0.95``.
        attr_sim_mat_beta: Weight on the attribute-similarity
            regularization loss. OpenEA's published value is ``0.001``.
        attr_max_epoch: Attr2Vec training epochs. OpenEA's published value
            is ``200``.
        sub_mat_size: Reference-pool-1 subsample size per
            attribute-similarity regularization step. OpenEA's published
            value is ``1000``.
        matching: Strategy used to resolve scored candidates into final
            links. Defaults to
            [GreedyMatcher][linkingtk.algorithms.matching.GreedyMatcher].
    """

    def __init__(
        self,
        embedding_dim: int = 100,
        num_epochs: int = 500,
        batch_size: int = 5000,
        learning_rate: float = 0.01,
        neg_alpha: float = 0.1,
        top_attr_threshold: float = 0.9,
        attr_sim_mat_threshold: float = 0.95,
        attr_sim_mat_beta: float = 0.001,
        attr_max_epoch: int = 200,
        sub_mat_size: int = 1000,
        matching: Matcher = DEFAULT_MATCHER,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.neg_alpha = neg_alpha
        self.top_attr_threshold = top_attr_threshold
        self.attr_sim_mat_threshold = attr_sim_mat_threshold
        self.attr_sim_mat_beta = attr_sim_mat_beta
        self.attr_max_epoch = attr_max_epoch
        self.sub_mat_size = sub_mat_size
        self.matching = matching
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
        attribute_triples1: list[Triple] | None = None,
        attribute_triples2: list[Triple] | None = None,
    ) -> JAPELinker:
        """Train a shared TransE space, optionally regularized by attribute similarity.

        Args:
            dataset1: Source entities -- also used to partition ``graph``
                into each KG's own triples (same approach as
                `IPTransELinker.fit`, for the same reason: negative
                sampling needs each KG's own entity pool).
            dataset2: Target entities. See ``dataset1``.
            ground_truth: List of ``(source_id, target_id)`` known-correct
                pairs. Seeds the shared embedding table (each pair's
                target is aliased to its source's id) and, if attribute
                triples are given, merges each pair's attribute
                vocabularies for Attr2Vec training.
            graph: The combined relational structure of both KGs (e.g.
                ``to_triples(graph1) + to_triples(graph2)``) -- entity ids
                on both sides must already be disjoint.
            random_state: Seed for reproducible training. Left
                unspecified, training is non-deterministic.
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
            attribute_triples1: KG1's ``(entity_id, predicate, value)``
                attribute triples, e.g. from a
                [GraphDatasetLoader][linkingtk.datasets.base.GraphDatasetLoader]
                that supports
                ``load_attribute_triples()``
                (see [linkingtk.datasets.openea_native][]). If ``None``
                (default) or empty, together with ``attribute_triples2``,
                training falls back to structural-only (see the module
                docstring's first deviation).
            attribute_triples2: KG2's attribute triples. See
                ``attribute_triples1``.

        Returns:
            ``self``, for chaining.

        Raises:
            LinkingTKError: If none of ``ground_truth``'s pairs have both
                ids present in ``dataset1``/``dataset2`` -- nothing to
                seed the shared embedding table with.
            OptionalDependencyError: If torch isn't installed.
        """
        try:
            import torch
            import torch.nn.functional as functional
        except ImportError as exc:
            raise OptionalDependencyError("JAPELinker", "kge") from exc

        if random_state is not None:
            torch.manual_seed(random_state)
        rng = np.random.default_rng(random_state)

        ids1 = {entity.id for entity in dataset1}
        ids2 = {entity.id for entity in dataset2}
        triples = to_triples(graph)
        triples1_labels = [t for t in triples if t[0] in ids1]
        triples2_labels = [t for t in triples if t[0] in ids2]

        seed_pairs = [(s, t) for s, t in ground_truth if s in ids1 and t in ids2]
        if not seed_pairs:
            raise LinkingTKError(
                "None of `ground_truth`'s pairs have both ids present in "
                "`dataset1`/`dataset2`; fit() has no seed pairs to train "
                "JAPE's shared embedding table with."
            )

        entity_to_id, relation_to_id = build_shared_id_mappings(
            triples1_labels + triples2_labels, seed_pairs
        )
        mapped1 = map_triples_to_ids(triples1_labels, entity_to_id, relation_to_id)
        mapped2 = map_triples_to_ids(triples2_labels, entity_to_id, relation_to_id)
        ctx1 = build_kg_context(mapped1)
        ctx2 = build_kg_context(mapped2)

        entity_embeds, relation_embeds = self._init_structural_embeddings(
            torch, functional, len(entity_to_id), len(relation_to_id)
        )
        optimizer = torch.optim.Adagrad([entity_embeds, relation_embeds], lr=self.learning_rate)
        sim_optimizer = torch.optim.Adagrad([entity_embeds], lr=self.learning_rate)

        attr_sim_mat, pool1_ids, pool2_ids = self._setup_attribute_pipeline(
            dataset1,
            dataset2,
            entity_to_id,
            seed_pairs,
            attribute_triples1,
            attribute_triples2,
            rng,
        )

        val_pairs = [
            (s, t) for s, t in (val_ground_truth or []) if s in entity_to_id and t in entity_to_id
        ]
        best_hits1 = -1.0
        epochs_without_improvement = 0

        for epoch in range(self.num_epochs):
            train_triple_epoch(
                entity_embeds,
                relation_embeds,
                optimizer,
                ctx1,
                ctx2,
                rng,
                self.batch_size,
                self.neg_alpha,
            )
            if attr_sim_mat is not None:
                train_sim_epoch(
                    entity_embeds,
                    sim_optimizer,
                    attr_sim_mat,
                    pool1_ids,
                    pool2_ids,
                    rng,
                    self.sub_mat_size,
                    self.attr_sim_mat_beta,
                )

            if val_pairs and (epoch + 1) % eval_every == 0:
                with torch.no_grad():
                    current_embeds = functional.normalize(entity_embeds, dim=1).numpy()
                hits1 = validation_hits1(current_embeds, entity_to_id, val_pairs)
                if hits1 <= best_hits1:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        break
                else:
                    best_hits1 = hits1
                    epochs_without_improvement = 0

        with torch.no_grad():
            final_embeds = functional.normalize(entity_embeds, dim=1).numpy()
        self._id_to_vector = {
            entity_id: final_embeds[index] for entity_id, index in entity_to_id.items()
        }
        self._fitted = True
        return self

    def _init_structural_embeddings(
        self, torch: Any, functional: Any, num_entities: int, num_relations: int
    ) -> tuple[Any, Any]:
        """Truncated-normal init (matches OpenEA's ``init="normal"``, same as `IPTransELinker`)."""
        std = 1.0 / math.sqrt(self.embedding_dim)

        def init(size: int) -> Any:
            return torch.nn.Parameter(
                functional.normalize(
                    torch.nn.init.trunc_normal_(
                        torch.empty(size, self.embedding_dim), std=std, a=-2 * std, b=2 * std
                    ),
                    dim=1,
                )
            )

        return init(num_entities), init(num_relations)

    def _setup_attribute_pipeline(
        self,
        dataset1: list[Entity],
        dataset2: list[Entity],
        entity_to_id: dict[str, int],
        seed_pairs: list[tuple[str, str]],
        attribute_triples1: list[Triple] | None,
        attribute_triples2: list[Triple] | None,
        rng: np.random.Generator,
    ) -> tuple[npt.NDArray[np.floating[Any]] | None, npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """Train Attr2Vec and build the thresholded attribute-similarity matrix.

        Returns ``(None, empty, empty)`` if no attribute triples are
        given, or if attribute/training-pair selection ends up empty --
        both are treated as "structural-only training", not an error.
        """
        empty = np.empty(0, dtype=np.int64)
        if not attribute_triples1 and not attribute_triples2:
            return None, empty, empty
        attribute_triples1 = attribute_triples1 or []
        attribute_triples2 = attribute_triples2 or []

        selected = select_popular_attributes(
            attribute_triples1, attribute_triples2, self.top_attr_threshold
        )
        entity_attrs = build_entity_attributes(
            attribute_triples1, attribute_triples2, selected, seed_pairs
        )
        training_pairs = generate_training_pairs(entity_attrs)
        if not training_pairs:
            return None, empty, empty

        attr_to_id = {attr: index for index, attr in enumerate(sorted(selected))}
        training_pairs_ids = np.array(
            [(attr_to_id[a], attr_to_id[b]) for a, b in training_pairs], dtype=np.int64
        )

        attr_embeds = train_attr2vec(
            training_pairs_ids,
            len(attr_to_id),
            self.embedding_dim,
            self.attr_max_epoch,
            self.batch_size,
            self.learning_rate,
            rng,
        )

        pool1_labels, pool2_labels = reference_pools(
            [entity.id for entity in dataset1 if entity.id in entity_to_id],
            [entity.id for entity in dataset2 if entity.id in entity_to_id],
            seed_pairs,
        )
        if not pool1_labels or not pool2_labels:
            return None, empty, empty

        vectors1 = pool_entity_attribute_vectors(
            pool1_labels, entity_attrs, attr_to_id, attr_embeds
        )
        vectors2 = pool_entity_attribute_vectors(
            pool2_labels, entity_attrs, attr_to_id, attr_embeds
        )
        attr_sim_mat = vectors1 @ vectors2.T
        attr_sim_mat[attr_sim_mat < self.attr_sim_mat_threshold] = 0.0
        attr_sim_mat = sparsify_to_row_argmax(attr_sim_mat)

        pool1_ids = np.array([entity_to_id[label] for label in pool1_labels], dtype=np.int64)
        pool2_ids = np.array([entity_to_id[label] for label in pool2_labels], dtype=np.int64)
        return attr_sim_mat, pool1_ids, pool2_ids

    def link(
        self,
        dataset1: list[Entity],
        dataset2: list[Entity],
        graph: Graph = None,
        blocking: BlockingStrategy = DEFAULT_BLOCKING,
    ) -> list[AlignmentResult]:
        if not self._fitted:
            raise LinkingTKError("JAPELinker.link() called before fit().")

        pairs = blocking.candidate_pairs(dataset1, dataset2)
        target_ids_by_source: dict[str, list[str]] = defaultdict(list)
        for entity1, entity2 in pairs:
            target_ids_by_source[entity1.id].append(entity2.id)

        candidates_by_source: dict[str, list[tuple[str, float]]] = {}
        for source_id, target_ids in target_ids_by_source.items():
            source_vector = self._embedding(source_id).reshape(1, -1)
            target_matrix = np.stack([self._embedding(target_id) for target_id in target_ids])
            scores = cosine_similarity(source_vector, target_matrix)[0]
            candidates_by_source[source_id] = list(
                zip(target_ids, (float(score) for score in scores), strict=True)
            )

        return self.matching.match(candidates_by_source)

    def _embedding(self, entity_id: str) -> npt.NDArray[np.floating[Any]]:
        vector = self._id_to_vector.get(entity_id)
        if vector is None:
            raise LinkingTKError(
                f"Entity {entity_id!r} has no trained embedding -- it didn't appear "
                "in fit()'s `graph`."
            )
        return vector
