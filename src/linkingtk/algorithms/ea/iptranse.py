"""IPTransE knowledge-graph-embedding linker for Entity Alignment.

Zhu, H., Xu, R., Yang, Y., Lin, X., & Cheng, X. (2017). Iterative Entity
Alignment via Joint Knowledge Embeddings. IJCAI 2017.
https://www.ijcai.org/proceedings/2017/0595.pdf

Like [MTransELinker][linkingtk.algorithms.ea.mtranse.MTransELinker], this is
a faithful reimplementation of IPTransE's actual training procedure, ported
directly from OpenEA's reference implementation
(https://github.com/nju-websoft/OpenEA/blob/master/src/openea/approaches/iptranse.py
and its ``modules/bootstrapping``, ``modules/train/batch.py``,
``modules/load/read.py`` support code) rather than from a from-scratch
reading of the paper. Three things distinguish it from MTransE:

- **Seed pairs share one embedding row**, not two rows plus a learned
  mapping (OpenEA's ``alignment_module="sharing"``): a seed pair's target
  entity is aliased to its source entity's id at training-tensor-build
  time. Alignment signal for held-out entities comes entirely from this
  shared space plus the two mechanisms below, not from any post-hoc
  projection.
- **Training loss is margin-based** (``margin=1.5``, one negative per
  positive, corrupting the head or tail from that triple's own KG), not
  MTransE's negative-sampling-free positive-only loss -- plus a
  relation-composition ("path") loss over 2-hop chains within each KG,
  weighted by ``path_parm``, that encourages composed relations
  (``r_x + r_y``) to approximate a direct edge (``r``) when both exist.
- **Iterative bootstrapping**: every ``bootstrap_every`` epochs, an
  unsupervised self-training round computes embedding similarity between
  every not-yet-seeded entity on each side, keeps each source entity's
  best match if it clears ``sim_th``, and turns each newly-found pair's
  real edges into weighted pseudo-triples trained for one extra epoch via
  a separate optimizer -- this is what "iterative" refers to in the
  method's name.

Ported into two private helper modules: ``_iptranse_training.py`` (plain
numpy, independently testable without ``torch``) and ``_iptranse_torch.py``
(the training-step functions that build/consume PyTorch tensors).

Deviations from OpenEA's own code, beyond the negative-sampling/path-generation
simplifications documented in those two modules:

- OpenEA partitions relation triples into two KGs via its own ``KGs``
  loader; this repo's EA linker family instead takes one *combined*
  ``graph`` in ``fit()``. IPTransE genuinely needs the KG1/KG2 boundary
  (per-KG negative sampling, path generation, pseudo-triple dicts), so
  it's re-derived here from ``dataset1``/``dataset2``'s own id sets: a
  triple belongs to KG1 iff its head id is one of ``dataset1``'s ids.
  This relies on the same assumption ``MTransELinker`` already documents
  (ids are disjoint across the two KGs) for a new purpose.
- OpenEA's bootstrap reference pool is ``valid_entities + test_entities``
  from its own loader (entities only, no labels needed). This repo's
  ``fit()`` only receives ``dataset1``/``dataset2`` (full entity rosters)
  and ``ground_truth`` (the train split) -- so the reference pool here is
  every ``dataset1``/``dataset2`` entity *not* already a ``ground_truth``
  source/target, which is broader than OpenEA's val+test-only pool. At
  real dataset scale this is a dense similarity matrix over most of both
  KGs' entities; ``bootstrap_pool_size`` lets callers cap it (random
  subsample, fixed for the whole run) if that's a problem.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from linkingtk.algorithms.base import DEFAULT_BLOCKING, BaseLinker
from linkingtk.algorithms.ea._iptranse_torch import (
    bootstrap_round,
    build_kg_context,
    train_epoch,
    validation_hits1,
)
from linkingtk.algorithms.ea._iptranse_training import build_shared_id_mappings
from linkingtk.algorithms.matching import DEFAULT_MATCHER, Matcher
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity
from linkingtk.core.result import AlignmentResult
from linkingtk.exceptions import LinkingTKError, OptionalDependencyError
from linkingtk.utils.graph import Graph, map_triples_to_ids, to_triples

if TYPE_CHECKING:
    import numpy.typing as npt


class IPTransELinker(BaseLinker):
    """Scores candidate pairs via a shared TransE space, path loss, and bootstrapping.

    Must be [fit][linkingtk.algorithms.ea.iptranse.IPTransELinker.fit]
    before [link][linkingtk.algorithms.base.BaseLinker.link] can be
    called. See the module docstring for how this differs from
    [MTransELinker][linkingtk.algorithms.ea.mtranse.MTransELinker].

    Args:
        embedding_dim: Dimensionality of the trained entity/relation
            embeddings. OpenEA's published EN-FR-15K-V1 config uses
            ``100``.
        num_epochs: Training epochs, each one pass over both KGs'
            triples (plus path-loss minibatches). OpenEA's config allows
            up to ``2000`` with early stopping (see ``val_ground_truth``
            on [fit][linkingtk.algorithms.ea.iptranse.IPTransELinker.fit]).
        batch_size: Mini-batch size for triple-loss training, split
            proportionally between the two KGs by triple count each
            epoch (matching OpenEA).
        learning_rate: Adagrad's learning rate, for both the main and
            bootstrap optimizers -- OpenEA's published value is ``0.01``.
        margin: Hinge margin for the triple and path losses. OpenEA's
            published value is ``1.5``.
        path_parm: Weight applied to the path (relation-composition)
            loss. OpenEA's published value is ``0.1``.
        sim_th: Minimum embedding similarity for a bootstrap round to
            accept a newly found pair. OpenEA's published value is
            ``0.7``.
        bootstrap_every: Run a bootstrapping round every this many
            epochs. OpenEA's published value (``bp_freq``) is ``100``.
        bootstrap_pool_size: Cap on each side's bootstrap reference pool
            size (random subsample, fixed for the whole run) -- see the
            module docstring's deviation note. ``None`` (default) uses
            the full pool.
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
        margin: float = 1.5,
        path_parm: float = 0.1,
        sim_th: float = 0.7,
        bootstrap_every: int = 100,
        bootstrap_pool_size: int | None = None,
        matching: Matcher = DEFAULT_MATCHER,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.margin = margin
        self.path_parm = path_parm
        self.sim_th = sim_th
        self.bootstrap_every = bootstrap_every
        self.bootstrap_pool_size = bootstrap_pool_size
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
    ) -> IPTransELinker:
        """Train a shared TransE space with margin+path loss and periodic bootstrapping.

        Args:
            dataset1: Source entities -- also used to partition ``graph``
                into each KG's own triples (see the module docstring).
            dataset2: Target entities. See ``dataset1``.
            ground_truth: List of ``(source_id, target_id)`` known-correct
                pairs. Seeds the shared embedding table (each pair's
                target is aliased to its source's id) and is excluded
                from the bootstrap reference pool.
            graph: The combined relational structure of both KGs (e.g.
                ``to_triples(graph1) + to_triples(graph2)`` from a
                [GraphDatasetLoader][linkingtk.datasets.base.GraphDatasetLoader]'s
                ``load_graphs()``) -- entity ids on both sides must
                already be disjoint, as they are from that loader.
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
            raise OptionalDependencyError("IPTransELinker", "kge") from exc

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
                "IPTransE's shared embedding table with."
            )

        entity_to_id, relation_to_id = build_shared_id_mappings(
            triples1_labels + triples2_labels, seed_pairs
        )
        mapped1 = map_triples_to_ids(triples1_labels, entity_to_id, relation_to_id)
        mapped2 = map_triples_to_ids(triples2_labels, entity_to_id, relation_to_id)
        ctx1 = build_kg_context(triples1_labels, mapped1, relation_to_id)
        ctx2 = build_kg_context(triples2_labels, mapped2, relation_to_id)

        pool1_ids, pool2_ids = self._bootstrap_pools(
            dataset1, dataset2, entity_to_id, seed_pairs, rng
        )
        combined_entity_pool = np.union1d(ctx1.entity_pool, ctx2.entity_pool)

        entity_embeds, relation_embeds = self._init_embeddings(
            torch, functional, len(entity_to_id), len(relation_to_id)
        )
        optimizer = torch.optim.Adagrad([entity_embeds, relation_embeds], lr=self.learning_rate)
        alignment_optimizer = torch.optim.Adagrad(
            [entity_embeds, relation_embeds], lr=self.learning_rate
        )

        val_pairs = [
            (s, t) for s, t in (val_ground_truth or []) if s in entity_to_id and t in entity_to_id
        ]
        best_hits1 = -1.0
        epochs_without_improvement = 0

        for epoch in range(self.num_epochs):
            train_epoch(
                entity_embeds,
                relation_embeds,
                optimizer,
                ctx1,
                ctx2,
                rng,
                self.batch_size,
                self.margin,
                self.path_parm,
            )

            if (epoch + 1) % self.bootstrap_every == 0:
                bootstrap_round(
                    entity_embeds,
                    relation_embeds,
                    alignment_optimizer,
                    ctx1,
                    ctx2,
                    pool1_ids,
                    pool2_ids,
                    combined_entity_pool,
                    rng,
                    self.batch_size,
                    self.margin,
                    self.sim_th,
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

    def _bootstrap_pools(
        self,
        dataset1: list[Entity],
        dataset2: list[Entity],
        entity_to_id: dict[str, int],
        seed_pairs: list[tuple[str, str]],
        rng: np.random.Generator,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """Every not-yet-seeded entity id on each side.

        See the module docstring's deviation note.
        """
        seed_sources = {entity_to_id[s] for s, _ in seed_pairs}
        seed_targets = {entity_to_id[t] for _, t in seed_pairs}
        pool1 = np.array(
            sorted({entity_to_id[e.id] for e in dataset1 if e.id in entity_to_id} - seed_sources),
            dtype=np.int64,
        )
        pool2 = np.array(
            sorted({entity_to_id[e.id] for e in dataset2 if e.id in entity_to_id} - seed_targets),
            dtype=np.int64,
        )
        if self.bootstrap_pool_size is not None:
            if len(pool1) > self.bootstrap_pool_size:
                pool1 = rng.choice(pool1, size=self.bootstrap_pool_size, replace=False)
            if len(pool2) > self.bootstrap_pool_size:
                pool2 = rng.choice(pool2, size=self.bootstrap_pool_size, replace=False)
        return pool1, pool2

    def _init_embeddings(
        self,
        torch: Any,
        functional: Any,
        num_entities: int,
        num_relations: int,
    ) -> tuple[Any, Any]:
        """Truncated-normal init.

        Matches OpenEA's ``init="normal"`` -- unlike
        [MTransELinker][linkingtk.algorithms.ea.mtranse.MTransELinker]'s
        ``"unit"`` init.
        """
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

    def link(
        self,
        dataset1: list[Entity],
        dataset2: list[Entity],
        graph: Graph = None,
        blocking: BlockingStrategy = DEFAULT_BLOCKING,
    ) -> list[AlignmentResult]:
        if not self._fitted:
            raise LinkingTKError("IPTransELinker.link() called before fit().")

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
