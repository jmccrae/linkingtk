"""SEA knowledge-graph-embedding linker for Entity Alignment.

Pei, S., Yu, L., Hoehndorf, R., & Zhang, X. (2019). Semi-Supervised Entity
Alignment via Knowledge Graph Embedding with Awareness of Degree
Difference. WWW 2019. https://dl.acm.org/citation.cfm?id=3313646

Like [MTransELinker][linkingtk.algorithms.ea.mtranse.MTransELinker], this is
a faithful reimplementation of SEA's actual training procedure, ported
directly from OpenEA's reference implementation
(https://github.com/nju-websoft/OpenEA/blob/master/src/openea/approaches/sea.py,
plus its ``models/basic_model.py`` and ``modules/train/batch.py`` support
code) rather than from a from-scratch reading of the paper.

SEA trains a **single shared entity/relation embedding table** with
**disjoint per-KG id ranges** (OpenEA's ``alignment_module="mapping"``, the
same shape [MTransELinker][linkingtk.algorithms.ea.mtranse.MTransELinker]
uses -- unlike [IPTransELinker][linkingtk.algorithms.ea.iptranse.IPTransELinker]'s
id-sharing or [BootEALinker][linkingtk.algorithms.ea.bootea.BootEALinker]'s
pseudo-triple-injecting "swapping"), via standard margin-based TransE with
uniform per-KG negative sampling -- no path loss, no bootstrapping, the
simplest structural half in this family after MTransE. What makes SEA
distinct is its mapping half:

- **Two** orthogonally-initialized square mapping matrices, ``mat_1``
  (KG1 -> KG2) and ``mat_2`` (KG2 -> KG1) -- not MTransE's single matrix.
- A **supervised** term over seed pairs: ``mat_1``-mapped source entities
  are pulled toward their real target (and symmetrically ``mat_2``-mapped
  targets toward their real source).
- A **semi-supervised (cycle-consistency)** term over *unlabeled* entities:
  a KG1 entity mapped through ``mat_1`` then back through ``mat_2`` should
  return to itself (and symmetrically for KG2) -- this needs no
  ground-truth pairing at all, just entity ids from both sides, which is
  what "semi-supervised" refers to in the method's name.

Ported into a private helper module, ``_sea_torch.py`` (the training-step
functions that build/consume PyTorch tensors; negative sampling itself
reuses ``_iptranse_training.py``'s ``sample_negative_triples`` directly,
since it's already fully generic).

**Deliberately not ported**: OpenEA's ``sea.py`` defines ``eye_mat_1``/
``eye_mat_2`` (the same orthogonality-regularizer constants
[MTransELinker][linkingtk.algorithms.ea.mtranse.MTransELinker]'s mapping
loss uses) but never references them in SEA's actual mapping loss -- no
orthogonality regularization is applied here, unlike MTransE. Same "port
actual, not apparent, reference behavior" situation as
[BootEALinker][linkingtk.algorithms.ea.bootea.BootEALinker]'s skipped
``likelihood()`` and [JAPELinker][linkingtk.algorithms.ea.jape.JAPELinker]'s
inert ``sim_optimizer`` (#28) -- skipped here entirely.

Deviations from OpenEA's own code:

- **``var_list=None`` gotcha, checked and applicable**: like
  [MTransELinker][linkingtk.algorithms.ea.mtranse.MTransELinker]/
  [KDCoELinker][linkingtk.algorithms.ea.kdcoe.KDCoELinker] (see
  ``feedback_openea_var_list_none``), OpenEA's
  ``generate_optimizer(mapping_loss, lr, opt='Adam')`` leaves ``var_list``
  unset, so TensorFlow differentiates the mapping loss against *every*
  trainable variable it touches -- both mapping matrices **and** the
  shared entity embeddings (every term reads from ``self.ent_embeds``).
  This port's mapping optimizer is scoped to
  ``[mapping_mat_1, mapping_mat_2, entity_embeds]``, not just "the two
  obvious" mapping matrices.
- **Optimizer is Adam**, not Adagrad -- unlike MTransE/IPTransE/BootEA.
  OpenEA's own ``SEA.init()`` asserts ``self.args.optimizer == 'Adam'``
  and its published ``sea_args_15K.json`` confirms it.
- **Unlabeled/semi-supervised pool**: OpenEA samples from its own loader's
  ``test_links + valid_links`` (using only entity ids, never the gold
  pairing). This repo's ``fit()`` only receives ``dataset1``/``dataset2``
  (full entity rosters) and ``ground_truth`` (the train split), so the
  pool here is every entity *not* already a ``ground_truth`` source/
  target, sampled independently per side -- the same broader-pool
  deviation already established for
  [IPTransELinker][linkingtk.algorithms.ea.iptranse.IPTransELinker]'s and
  [BootEALinker][linkingtk.algorithms.ea.bootea.BootEALinker]'s bootstrap
  pools, and in practice covers the same entities OpenEA's own pool would
  (val+test).
- **KG1/KG2 triple partition**: like
  [IPTransELinker][linkingtk.algorithms.ea.iptranse.IPTransELinker], this
  repo takes one *combined* ``graph`` in ``fit()`` rather than OpenEA's
  own separately-loaded per-KG ``KGs``; a triple belongs to KG1 iff its
  head id is one of ``dataset1``'s ids, relying on the same disjoint-id
  assumption ``MTransELinker`` already documents.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from linkingtk.algorithms.base import DEFAULT_BLOCKING, BaseLinker
from linkingtk.algorithms.ea._device import resolve_device
from linkingtk.algorithms.ea._sea_torch import (
    build_kg_context,
    train_mapping_epoch,
    train_structural_epoch,
    validation_hits1,
)
from linkingtk.algorithms.matching import DEFAULT_MATCHER, Matcher
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity
from linkingtk.core.result import AlignmentResult
from linkingtk.exceptions import LinkingTKError, OptionalDependencyError
from linkingtk.utils.graph import Graph, build_id_mappings, map_triples_to_ids, to_triples

if TYPE_CHECKING:
    import numpy.typing as npt


class SEALinker(BaseLinker):
    """Scores candidate pairs via structural TransE plus a semi-supervised dual mapping.

    Must be [fit][linkingtk.algorithms.ea.sea.SEALinker.fit] before
    [link][linkingtk.algorithms.base.BaseLinker.link] can be called.

    Args:
        embedding_dim: Dimensionality of the trained entity/relation
            embeddings and both (square) mapping matrices. OpenEA's
            published EN-FR-15K-V1 config uses ``100``.
        num_epochs: Training epochs, each alternating one epoch of
            structural triple-loss training with one epoch of mapping-loss
            training. OpenEA's config allows up to ``2000`` with early
            stopping (see ``val_ground_truth`` on
            [fit][linkingtk.algorithms.ea.sea.SEALinker.fit]).
        batch_size: Mini-batch size for structural triple-loss training,
            split proportionally between the two KGs by triple count each
            epoch. The mapping loss reuses the resulting step count.
        learning_rate: Adam's learning rate, for both the structural and
            mapping optimizers -- OpenEA's published value is ``0.01``.
        margin: Hinge margin for the structural triple loss. OpenEA's
            published value is ``1.5``.
        alpha_1: Weight applied to the supervised mapping loss (seed-pair
            approximation error). OpenEA's published value is ``2.5``.
        alpha_2: Weight applied to the semi-supervised (cycle-consistency)
            mapping loss over unlabeled entities. OpenEA's published value
            is ``0.25``.
        matching: Strategy used to resolve scored candidates into final
            links. Defaults to
            [GreedyMatcher][linkingtk.algorithms.matching.GreedyMatcher].
        device: Torch device to train on, e.g. ``"cpu"`` (default) or
            ``"cuda"``/``"cuda:0"``. Trained embeddings are always stored
            as CPU numpy arrays regardless of this setting.
    """

    def __init__(
        self,
        embedding_dim: int = 100,
        num_epochs: int = 500,
        batch_size: int = 5000,
        learning_rate: float = 0.01,
        margin: float = 1.5,
        alpha_1: float = 2.5,
        alpha_2: float = 0.25,
        matching: Matcher = DEFAULT_MATCHER,
        device: str = "cpu",
    ) -> None:
        self.embedding_dim = embedding_dim
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.margin = margin
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.matching = matching
        self.device = device
        self._id_to_vector: dict[str, npt.NDArray[np.floating[Any]]] = {}
        self._mapping: npt.NDArray[np.floating[Any]] | None = None
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
    ) -> SEALinker:
        """Train shared structural TransE embeddings plus a dual cross-KG mapping.

        Args:
            dataset1: Source entities -- also used to partition ``graph``
                into each KG's own triples for structural negative
                sampling and to build the semi-supervised unlabeled pool
                (see the module docstring).
            dataset2: Target entities. See ``dataset1``.
            ground_truth: List of ``(source_id, target_id)`` known-correct
                pairs used to train the supervised half of the mapping
                loss. Unlike ``KGELinker``, these are *not* injected into
                ``graph`` as extra triples.
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
                ids present in ``graph``'s own triples -- nothing to train
                the mapping with -- or if ``device`` is invalid or
                unavailable.
            OptionalDependencyError: If torch isn't installed.
        """
        try:
            import torch
            import torch.nn.functional as functional
        except ImportError as exc:
            raise OptionalDependencyError("SEALinker", "kge") from exc

        device = resolve_device(self.device)
        if random_state is not None:
            torch.manual_seed(random_state)
            torch.cuda.manual_seed_all(random_state)
        rng = np.random.default_rng(random_state)

        ids1 = {entity.id for entity in dataset1}
        ids2 = {entity.id for entity in dataset2}
        triples = to_triples(graph)
        triples1_labels = [t for t in triples if t[0] in ids1]
        triples2_labels = [t for t in triples if t[0] in ids2]

        entity_to_id, relation_to_id = build_id_mappings(triples)
        mapped1 = map_triples_to_ids(triples1_labels, entity_to_id, relation_to_id)
        mapped2 = map_triples_to_ids(triples2_labels, entity_to_id, relation_to_id)
        ctx1 = build_kg_context(triples1_labels, mapped1)
        ctx2 = build_kg_context(triples2_labels, mapped2)

        seed_pairs = [(s, t) for s, t in ground_truth if s in entity_to_id and t in entity_to_id]
        if not seed_pairs:
            raise LinkingTKError(
                "None of `ground_truth`'s pairs have both ids present in `graph`'s "
                "own triples; fit() has no seed pairs to train SEA's mapping with."
            )
        seed_source = torch.tensor([entity_to_id[s] for s, _ in seed_pairs], device=device)
        seed_target = torch.tensor([entity_to_id[t] for _, t in seed_pairs], device=device)

        seed_sources = {s for s, _ in seed_pairs}
        seed_targets = {t for _, t in seed_pairs}
        unlabeled_pool1 = np.array(
            sorted({entity_to_id[e.id] for e in dataset1 if e.id in entity_to_id} - seed_sources),
            dtype=np.int64,
        )
        unlabeled_pool2 = np.array(
            sorted({entity_to_id[e.id] for e in dataset2 if e.id in entity_to_id} - seed_targets),
            dtype=np.int64,
        )

        entity_embeds, relation_embeds = self._init_structural_embeddings(
            torch, functional, len(entity_to_id), len(relation_to_id), device
        )
        mapping_shape = (self.embedding_dim, self.embedding_dim)
        mapping_mat_1 = torch.nn.Parameter(
            torch.nn.init.orthogonal_(torch.empty(*mapping_shape, device=device))
        )
        mapping_mat_2 = torch.nn.Parameter(
            torch.nn.init.orthogonal_(torch.empty(*mapping_shape, device=device))
        )

        triple_optimizer = torch.optim.Adam([entity_embeds, relation_embeds], lr=self.learning_rate)
        # `entity_embeds` is included here (not just the two mapping
        # matrices) to match OpenEA's `generate_optimizer(mapping_loss,
        # lr, var_list=None, opt='Adam')` -- see the module docstring.
        mapping_optimizer = torch.optim.Adam(
            [mapping_mat_1, mapping_mat_2, entity_embeds], lr=self.learning_rate
        )

        total_triples = len(ctx1.triples) + len(ctx2.triples)
        triple_steps = max(1, (total_triples + self.batch_size - 1) // self.batch_size)

        val_pairs = [
            (s, t) for s, t in (val_ground_truth or []) if s in entity_to_id and t in entity_to_id
        ]
        best_hits1 = -1.0
        epochs_without_improvement = 0

        for epoch in range(self.num_epochs):
            train_structural_epoch(
                entity_embeds,
                relation_embeds,
                triple_optimizer,
                ctx1,
                ctx2,
                rng,
                self.batch_size,
                self.margin,
            )
            train_mapping_epoch(
                entity_embeds,
                mapping_mat_1,
                mapping_mat_2,
                mapping_optimizer,
                seed_source,
                seed_target,
                unlabeled_pool1,
                unlabeled_pool2,
                rng,
                triple_steps,
                self.alpha_1,
                self.alpha_2,
            )

            if val_pairs and (epoch + 1) % eval_every == 0:
                with torch.no_grad():
                    current_embeds = functional.normalize(entity_embeds, dim=1).cpu().numpy()
                    current_mapping = mapping_mat_1.cpu().numpy()
                hits1 = validation_hits1(current_embeds, current_mapping, entity_to_id, val_pairs)
                if hits1 <= best_hits1:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        break
                else:
                    best_hits1 = hits1
                    epochs_without_improvement = 0

        with torch.no_grad():
            final_embeds = functional.normalize(entity_embeds, dim=1).cpu().numpy()
            final_mapping = mapping_mat_1.cpu().numpy()
        self._id_to_vector = {
            entity_id: final_embeds[index] for entity_id, index in entity_to_id.items()
        }
        self._mapping = final_mapping
        self._fitted = True
        return self

    def _init_structural_embeddings(
        self,
        torch: Any,
        functional: Any,
        num_entities: int,
        num_relations: int,
        device: Any,
    ) -> tuple[Any, Any]:
        """Truncated-normal init, matching OpenEA's ``init="normal"``.

        Same shape as
        [IPTransELinker][linkingtk.algorithms.ea.iptranse.IPTransELinker]'s
        ``_init_embeddings`` -- unlike
        [MTransELinker][linkingtk.algorithms.ea.mtranse.MTransELinker]'s
        ``"unit"`` init.
        """
        import math

        std = 1.0 / math.sqrt(self.embedding_dim)

        def init(size: int) -> Any:
            return torch.nn.Parameter(
                functional.normalize(
                    torch.nn.init.trunc_normal_(
                        torch.empty(size, self.embedding_dim, device=device),
                        std=std,
                        a=-2 * std,
                        b=2 * std,
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
            raise LinkingTKError("SEALinker.link() called before fit().")

        pairs = blocking.candidate_pairs(dataset1, dataset2)
        target_ids_by_source: dict[str, list[str]] = defaultdict(list)
        for entity1, entity2 in pairs:
            target_ids_by_source[entity1.id].append(entity2.id)

        candidates_by_source: dict[str, list[tuple[str, float]]] = {}
        for source_id, target_ids in target_ids_by_source.items():
            source_vector = self.source_embedding(source_id).reshape(1, -1)
            target_matrix = np.stack([self.target_embedding(target_id) for target_id in target_ids])
            scores = cosine_similarity(source_vector, target_matrix)[0]
            candidates_by_source[source_id] = list(
                zip(target_ids, (float(score) for score in scores), strict=True)
            )

        return self.matching.match(candidates_by_source)

    def source_embedding(self, entity_id: str) -> npt.NDArray[np.floating[Any]]:
        """Vector used to score ``entity_id`` as a scored pair's source side.

        Projected via ``mapping_mat_1``.
        """
        return self._project(self._embedding(entity_id))

    def target_embedding(self, entity_id: str) -> npt.NDArray[np.floating[Any]]:
        """Vector used to score ``entity_id`` as a scored pair's target side (unprojected)."""
        return self._embedding(entity_id)

    def _embedding(self, entity_id: str) -> npt.NDArray[np.floating[Any]]:
        vector = self._id_to_vector.get(entity_id)
        if vector is None:
            raise LinkingTKError(
                f"Entity {entity_id!r} has no trained embedding -- it didn't appear "
                "in fit()'s `graph`."
            )
        return vector

    def _project(self, vector: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
        assert self._mapping is not None  # noqa: S101 -- guarded by self._fitted in link()
        return vector @ self._mapping
