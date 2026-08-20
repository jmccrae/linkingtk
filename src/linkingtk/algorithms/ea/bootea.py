"""BootEA knowledge-graph-embedding linker for Entity Alignment.

Sun, Z., Hu, W., Zhang, Q., & Qu, Y. (2018). Bootstrapping Entity
Alignment with Knowledge Graph Embedding. IJCAI 2018.
https://www.ijcai.org/proceedings/2018/0611.pdf

Like [MTransELinker][linkingtk.algorithms.ea.mtranse.MTransELinker] and its
siblings, this is a faithful reimplementation of BootEA's actual training
procedure, ported directly from OpenEA's reference implementation
(https://github.com/nju-websoft/OpenEA/blob/master/src/openea/approaches/bootea.py,
which extends ``approaches/aligne.py``, plus its
``modules/bootstrapping/alignment_finder.py``,
``modules/train/batch.py``, ``modules/load/kgs.py`` support code) rather
than from a from-scratch reading of the paper.

BootEA trains a **single shared entity/relation embedding table**
(disjoint per-KG id ranges, no id merging -- OpenEA's
``alignment_module="swapping"``, unlike
[IPTransELinker][linkingtk.algorithms.ea.iptranse.IPTransELinker]'s
"sharing" id-merge and unlike
[MTransELinker][linkingtk.algorithms.ea.mtranse.MTransELinker]'s learned
mapping matrix) with the **"limited" loss** -- a double-margin hinge
(``pos_loss = relu(pos_score - pos_margin)``,
``neg_loss = relu(neg_margin - neg_score)``), not the single-hinge margin
loss MTransE/IPTransE use. Three mechanisms distinguish it from a plain
structural baseline:

1. **Seed pairs are baked into the main structural triple set from the
   start**, not fed through a separate loss channel: for each seed pair
   ``(a, b)``, every real KG1 triple touching ``a`` becomes an extra
   pseudo-triple with ``a`` replaced by ``b`` (and symmetrically for KG2),
   folded directly into each side's own triple set (OpenEA's
   ``generate_sup_relation_triples``, called once at ``KGs``-construction
   time) -- so the main loss pulls ``a``'s and ``b``'s embeddings together
   implicitly via shared structural neighbors, no mapping matrix needed.
2. **Truncated (hard) negative sampling**: each triple's corrupted
   head/tail is drawn from that entity's current K-nearest-neighbor set by
   embedding similarity (only the top ``1 - truncated_epsilon`` fraction
   of same-KG entities are eligible), not uniformly -- refreshed every
   outer iteration from the live embeddings.
3. **Bootstrapping with two-sided editing + max-weight bipartite
   matching**, run once per outer iteration (every ``sub_epoch`` epochs):
   candidate pairs are filtered by similarity threshold *and* row-wise
   top-``k``, then resolved into a true 1-to-1 assignment via maximum-
   weight bipartite matching (see
   [_bootea_training][linkingtk.algorithms.ea._bootea_training] for the
   ``scipy``-based substitution for OpenEA's ``graph_tool``/``igraph``),
   then merged into an accumulated labeled-alignment set with editing
   (a later, higher-confidence match can overwrite an earlier one). The
   *entire* current labeled-alignment set -- not just this round's deltas
   -- is turned into pseudo-triples the same way seed pairs are, and
   trained for exactly 1 epoch through a **separate** alignment
   loss/optimizer (a positive-only sigmoid loss, no negative sampling --
   a different shape than the main limited loss).

**Deliberately not ported**: OpenEA's ``bootea.py`` also defines a
``likelihood()`` EM-style step (``_define_likelihood_graph``), but
``run()``'s call to it is commented out
(``# self.likelihood(labeled_align)``) -- the reference never actually
runs it. Same "port the actual, not apparent, reference behavior"
situation as [JAPELinker][linkingtk.algorithms.ea.jape.JAPELinker]'s
``sim_optimizer`` (#28) -- skipped here entirely.

Ported into two private helper modules:
``_bootea_training.py`` (plain numpy, independently testable without
``torch``) and ``_bootea_torch.py`` (the training-step functions that
build/consume PyTorch tensors).

Deviations from OpenEA's own code, beyond what those two modules already
document:

- **Bootstrap reference pool**: OpenEA restricts the "not yet aligned"
  search pool to its own dataset loader's ``valid_entities + test_entities``.
  This repo's ``fit()`` only receives ``dataset1``/``dataset2`` and
  ``ground_truth`` (the train split), so the pool here is every entity
  *not* already a ``ground_truth`` source/target -- broader than OpenEA's
  pool, same already-established deviation as
  [IPTransELinker][linkingtk.algorithms.ea.iptranse.IPTransELinker]'s
  bootstrap pool and
  [JAPELinker][linkingtk.algorithms.ea.jape.JAPELinker]'s attribute
  reference pool (reuses ``_jape_training.py``'s ``reference_pools``
  directly).
- **Max-weight bipartite matching via ``scipy.optimize.linear_sum_assignment``**
  instead of OpenEA's ``graph_tool``/``igraph`` -- see
  ``_bootea_training.py``'s ``find_mwgm_pairs`` docstring.
- **``var_list=None`` gotcha checked, not applicable**: unlike
  MTransE/KDCoE, BootEA has no separate mapping-matrix parameter to
  under-scope an optimizer against -- ``ent_embeds``/``rel_embeds`` are
  the only trainable variables in scope for every optimizer this method
  defines, so leaving an optimizer's variable list unset already resolves
  to exactly the intended target.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from linkingtk.algorithms.base import DEFAULT_BLOCKING, BaseLinker
from linkingtk.algorithms.ea._bootea_torch import (
    build_kg_context,
    train_alignment_epoch,
    train_structural_epoch,
    validation_hits1,
)
from linkingtk.algorithms.ea._bootea_training import (
    compute_truncated_neighbors,
    edit_labeled_alignment,
    find_mwgm_pairs,
    pseudo_triples_for_pairs,
)
from linkingtk.algorithms.ea._jape_training import reference_pools
from linkingtk.algorithms.matching import DEFAULT_MATCHER, Matcher
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity
from linkingtk.core.result import AlignmentResult
from linkingtk.core.source import EntitySource
from linkingtk.exceptions import LinkingTKError, OptionalDependencyError
from linkingtk.utils.device import resolve_device
from linkingtk.utils.graph import Graph, build_id_mappings, map_triples_to_ids, to_triples

if TYPE_CHECKING:
    import numpy.typing as npt


class BootEALinker(BaseLinker):
    """Scores candidate pairs via a shared TransE space, grown by editing-based bootstrapping.

    Must be [fit][linkingtk.algorithms.ea.bootea.BootEALinker.fit] before
    [link][linkingtk.algorithms.base.BaseLinker.link] can be called. See
    the module docstring for how this differs from
    [MTransELinker][linkingtk.algorithms.ea.mtranse.MTransELinker] and
    [IPTransELinker][linkingtk.algorithms.ea.iptranse.IPTransELinker].

    Args:
        embedding_dim: Dimensionality of the trained entity/relation
            embeddings. OpenEA's published EN-FR-15K-V1 config uses
            ``100``.
        num_epochs: Total structural-training-epoch cap, split across
            outer bootstrapping iterations of ``sub_epoch`` epochs each.
            OpenEA's config allows up to ``2000`` with early stopping (see
            ``val_ground_truth`` on
            [fit][linkingtk.algorithms.ea.bootea.BootEALinker.fit]).
        sub_epoch: Epochs per outer iteration, between bootstrapping
            rounds. OpenEA's published value is ``10``.
        batch_size: Mini-batch size for both the structural and alignment
            losses. OpenEA's published value is ``5000``.
        learning_rate: Adagrad's learning rate, for both optimizers.
            OpenEA's published value is ``0.01``.
        pos_margin: Positive-triple margin in the "limited" loss --
            positive scores aren't penalized until they exceed this.
            OpenEA's published value is ``0.01``.
        neg_margin: Negative-triple margin -- negative scores aren't
            penalized until they drop below this. OpenEA's published
            value is ``2.0``.
        neg_margin_balance: Weight applied to the negative-triple loss
            term relative to the positive term. OpenEA's published value
            is ``0.2``.
        neg_triple_num: Negative triples sampled per positive. OpenEA's
            published value is ``10``.
        truncated_epsilon: Fraction of each KG's entities *excluded* from
            an entity's truncated negative-sampling candidate set (only
            the top ``1 - truncated_epsilon`` most similar entities are
            eligible). OpenEA's published value is ``0.9``.
        sim_th: Minimum similarity for a bootstrap round's candidate pair.
            OpenEA's published value is ``0.7``.
        k: Max candidate matches kept per reference-pool row before
            max-weight bipartite matching. OpenEA's published value is
            ``10``.
        bootstrap_pool_size: Cap on each side's bootstrap reference pool
            size (random subsample, fixed for the whole run) -- same
            deviation-mitigation as
            [IPTransELinker][linkingtk.algorithms.ea.iptranse.IPTransELinker]'s
            own parameter: this repo's broader-than-OpenEA reference pool
            (see the module docstring) means ``find_mwgm_pairs``'s
            ``scipy.optimize.linear_sum_assignment`` step runs over a
            denser candidate set than OpenEA's own val+test-only pool
            would produce. ``None`` (default) uses the full pool.
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
        num_epochs: int = 2000,
        sub_epoch: int = 10,
        batch_size: int = 5000,
        learning_rate: float = 0.01,
        pos_margin: float = 0.01,
        neg_margin: float = 2.0,
        neg_margin_balance: float = 0.2,
        neg_triple_num: int = 10,
        truncated_epsilon: float = 0.9,
        sim_th: float = 0.7,
        k: int = 10,
        bootstrap_pool_size: int | None = None,
        matching: Matcher = DEFAULT_MATCHER,
        device: str = "cpu",
    ) -> None:
        self.embedding_dim = embedding_dim
        self.num_epochs = num_epochs
        self.sub_epoch = sub_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.neg_margin_balance = neg_margin_balance
        self.neg_triple_num = neg_triple_num
        self.truncated_epsilon = truncated_epsilon
        self.sim_th = sim_th
        self.k = k
        self.bootstrap_pool_size = bootstrap_pool_size
        self.matching = matching
        self.device = device
        self._id_to_vector: dict[str, npt.NDArray[np.floating[Any]]] = {}
        self._fitted = False

    def fit(  # noqa: PLR0915 -- the bootstrapping loop genuinely needs this much sequential setup
        self,
        dataset1: list[Entity],
        dataset2: list[Entity],
        ground_truth: list[tuple[str, str]],
        graph: Graph,
        random_state: int | None = None,
        val_ground_truth: list[tuple[str, str]] | None = None,
        patience: int = 5,
        eval_every: int = 10,
    ) -> BootEALinker:
        """Train a shared TransE space with seed pseudo-triples and iterative bootstrapping.

        See the module docstring for the three mechanisms that
        distinguish this from a plain structural baseline.

        Args:
            dataset1: Source entities -- also used to partition ``graph``
                into each KG's own triples (same approach
                [IPTransELinker][linkingtk.algorithms.ea.iptranse.IPTransELinker]
                uses) and as the bootstrap reference pool's source.
            dataset2: Target entities. See ``dataset1``.
            ground_truth: List of ``(source_id, target_id)`` known-correct
                pairs. Seeds the main structural triple set (via pseudo-
                triples) and is excluded from the bootstrap reference pool.
            graph: The combined relational structure of both KGs -- entity
                ids on both sides must already be disjoint.
            random_state: Seed for reproducible training. Left
                unspecified, training is non-deterministic.
            val_ground_truth: Optional held-out pairs used for early
                stopping -- every ``eval_every`` epochs, Hits@1 is checked,
                and training stops after ``patience`` checks with no
                improvement (no further bootstrapping rounds run once
                stopped, matching OpenEA's own early-stop-before-
                bootstrap ordering).
            patience: Number of non-improving ``eval_every``-spaced checks
                to tolerate before stopping early. Only used if
                ``val_ground_truth`` is given.
            eval_every: How often (in epochs) to check ``val_ground_truth``.
                Only used if ``val_ground_truth`` is given.

        Returns:
            ``self``, for chaining.

        Raises:
            LinkingTKError: If none of ``ground_truth``'s pairs have both
                ids present in ``graph``'s own triples -- nothing to seed
                the structural triple set with -- or if ``device`` is
                invalid or unavailable.
            OptionalDependencyError: If torch isn't installed.
        """
        try:
            import torch
            import torch.nn.functional as functional
        except ImportError as exc:
            raise OptionalDependencyError("BootEALinker", "kge") from exc

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

        entity_to_id, relation_to_id = build_id_mappings(triples1_labels + triples2_labels)
        mapped1 = map_triples_to_ids(triples1_labels, entity_to_id, relation_to_id)
        mapped2 = map_triples_to_ids(triples2_labels, entity_to_id, relation_to_id)

        seed_pairs = [(s, t) for s, t in ground_truth if s in entity_to_id and t in entity_to_id]
        if not seed_pairs:
            raise LinkingTKError(
                "None of `ground_truth`'s pairs have both ids present in `graph`'s "
                "own triples; fit() has no seed pairs to train BootEA's structural "
                "triple set with."
            )
        seed_pair_ids = [(entity_to_id[s], entity_to_id[t]) for s, t in seed_pairs]

        real_ctx1 = build_kg_context(mapped1)
        real_ctx2 = build_kg_context(mapped2)
        seed_triples1, seed_triples2 = pseudo_triples_for_pairs(
            seed_pair_ids,
            real_ctx1.by_head,
            real_ctx1.by_tail,
            real_ctx2.by_head,
            real_ctx2.by_tail,
        )
        combined1 = self._augment_triples(mapped1, seed_triples1)
        combined2 = self._augment_triples(mapped2, seed_triples2)
        ctx1 = build_kg_context(combined1)
        ctx2 = build_kg_context(combined2)

        kg1_entity_ids = np.array(
            sorted(entity_to_id[e] for e in ids1 if e in entity_to_id), dtype=np.int64
        )
        kg2_entity_ids = np.array(
            sorted(entity_to_id[e] for e in ids2 if e in entity_to_id), dtype=np.int64
        )
        neighbor_k1 = max(1, int(round((1 - self.truncated_epsilon) * len(kg1_entity_ids))))
        neighbor_k2 = max(1, int(round((1 - self.truncated_epsilon) * len(kg2_entity_ids))))

        entity_embeds, relation_embeds = self._init_embeddings(
            torch, functional, len(entity_to_id), len(relation_to_id), device
        )
        structural_optimizer = torch.optim.Adagrad(
            [entity_embeds, relation_embeds], lr=self.learning_rate
        )
        alignment_optimizer = torch.optim.Adagrad(
            [entity_embeds, relation_embeds], lr=self.learning_rate
        )

        val_pairs = [
            (s, t) for s, t in (val_ground_truth or []) if s in entity_to_id and t in entity_to_id
        ]

        pool1_labels, pool2_labels = reference_pools(sorted(ids1), sorted(ids2), seed_pairs)
        pool1_ids = np.array([entity_to_id[e] for e in pool1_labels], dtype=np.int64)
        pool2_ids = np.array([entity_to_id[e] for e in pool2_labels], dtype=np.int64)
        if self.bootstrap_pool_size is not None:
            if len(pool1_ids) > self.bootstrap_pool_size:
                pool1_ids = rng.choice(pool1_ids, size=self.bootstrap_pool_size, replace=False)
            if len(pool2_ids) > self.bootstrap_pool_size:
                pool2_ids = rng.choice(pool2_ids, size=self.bootstrap_pool_size, replace=False)

        labeled_alignment: dict[int, int] = {}
        neighbors1: dict[int, npt.NDArray[np.int64]] = {}
        neighbors2: dict[int, npt.NDArray[np.int64]] = {}

        num_outer_iters = max(1, self.num_epochs // self.sub_epoch)
        best_hits1 = -1.0
        epochs_without_improvement = 0
        epoch = 0
        stopped_early = False

        for _outer in range(num_outer_iters):
            for _ in range(self.sub_epoch):
                train_structural_epoch(
                    entity_embeds,
                    relation_embeds,
                    structural_optimizer,
                    ctx1,
                    ctx2,
                    neighbors1,
                    neighbors2,
                    rng,
                    self.batch_size,
                    self.pos_margin,
                    self.neg_margin,
                    self.neg_margin_balance,
                    self.neg_triple_num,
                )
                epoch += 1

                if val_pairs and epoch % eval_every == 0:
                    with torch.no_grad():
                        current_embeds = functional.normalize(entity_embeds, dim=1).cpu().numpy()
                    hits1 = validation_hits1(current_embeds, entity_to_id, val_pairs)
                    if hits1 <= best_hits1:
                        epochs_without_improvement += 1
                        if epochs_without_improvement >= patience:
                            stopped_early = True
                            break
                    else:
                        best_hits1 = hits1
                        epochs_without_improvement = 0

            if stopped_early:
                break

            if len(pool1_ids) and len(pool2_ids):
                with torch.no_grad():
                    current = functional.normalize(entity_embeds, dim=1).cpu().numpy()
                sim_mat = current[pool1_ids] @ current[pool2_ids].T
                curr_matches = find_mwgm_pairs(sim_mat, self.sim_th, self.k)
                labeled_alignment = edit_labeled_alignment(
                    labeled_alignment, {(i, j) for i, j, _ in curr_matches}, sim_mat
                )

                if labeled_alignment:
                    aligned_pairs = [
                        (int(pool1_ids[i]), int(pool2_ids[j])) for i, j in labeled_alignment.items()
                    ]
                    align_triples1, align_triples2 = pseudo_triples_for_pairs(
                        aligned_pairs, ctx1.by_head, ctx1.by_tail, ctx2.by_head, ctx2.by_tail
                    )
                    train_alignment_epoch(
                        entity_embeds,
                        relation_embeds,
                        alignment_optimizer,
                        np.array(align_triples1, dtype=np.int64).reshape(-1, 3),
                        np.array(align_triples2, dtype=np.int64).reshape(-1, 3),
                        self.batch_size,
                    )

            with torch.no_grad():
                current = functional.normalize(entity_embeds, dim=1).cpu().numpy()
            neighbors1 = compute_truncated_neighbors(current, kg1_entity_ids, neighbor_k1)
            neighbors2 = compute_truncated_neighbors(current, kg2_entity_ids, neighbor_k2)

        with torch.no_grad():
            final_embeds = functional.normalize(entity_embeds, dim=1).cpu().numpy()
        self._id_to_vector = {
            entity_id: final_embeds[index] for entity_id, index in entity_to_id.items()
        }
        self._fitted = True
        return self

    @staticmethod
    def _augment_triples(
        mapped: npt.NDArray[np.int64], extra: list[tuple[int, int, int]]
    ) -> npt.NDArray[np.int64]:
        if not extra:
            return mapped
        extra_array = np.array(extra, dtype=np.int64).reshape(-1, 3)
        return np.concatenate([mapped, extra_array], axis=0)

    def _init_embeddings(
        self,
        torch: Any,
        functional: Any,
        num_entities: int,
        num_relations: int,
        device: Any,
    ) -> tuple[Any, Any]:
        """Truncated-normal init, matching OpenEA's ``init="normal"`` (same as IPTransE's)."""
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
        dataset2: list[Entity] | EntitySource,
        graph: Graph = None,
        blocking: BlockingStrategy = DEFAULT_BLOCKING,
    ) -> list[AlignmentResult]:
        if not self._fitted:
            raise LinkingTKError("BootEALinker.link() called before fit().")

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
