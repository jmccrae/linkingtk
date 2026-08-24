"""GCN-Align knowledge-graph-embedding linker for Entity Alignment.

Wang, Z., Lv, Q., Lan, X., & Zhang, Y. (2018). Cross-lingual Knowledge
Graph Alignment via Graph Convolutional Networks. EMNLP 2018.
https://aclanthology.org/D18-1032/

The first GNN-based EA linker in this package (see #18's parent issue and
DESIGN.md's Entity Alignment references) -- unlike every linker in
``linkingtk.algorithms.ea`` so far, entity representations come from
propagating embeddings through a fixed graph convolution over the combined
relational structure, not from a triple-local translational/distance loss
over embeddings directly. This is a faithful port of OpenEA's reference
implementation
(https://github.com/nju-websoft/OpenEA/blob/master/src/openea/approaches/gcn_align.py),
not a from-scratch reading of the paper, following this repo's established
convention (e.g. [MTransELinker][linkingtk.algorithms.ea.mtranse.MTransELinker],
[RSN4EALinker][linkingtk.algorithms.ea.rsn4ea.RSN4EALinker]).

**Both of OpenEA's branches are implemented**: the structural (``se``)
branch propagates a directly-learnable per-entity embedding table; the
attribute (``ae``) branch propagates a learned transform of each entity's
one-hot attribute-*predicate*-presence vector (see
[build_attribute_features][linkingtk.algorithms.ea._gcn_align_training.build_attribute_features]
for why predicates, not values). Both are 2-layer GCNs over the *same*
structural adjacency, trained independently (separate optimizers, never a
combined loss) with the *same* shared negatives each step, matching
OpenEA's own ``GCN_Align.train_embeddings``. At evaluation time (and for
early-stopping/final embeddings), scores use the concatenation
``[se * beta, ae * (1 - beta)]`` -- OpenEA's own ``test_method: "sa"``,
the config that produced the published EN-FR-15K-V1 numbers.
``use_attributes=False`` (default) skips the ``ae`` branch entirely,
needing nothing beyond [EnFr15KDataset][linkingtk.datasets.EnFr15KDataset]'s
plain relational triples -- pass ``use_attributes=True`` plus
[EnFr15KAttrDataset][linkingtk.datasets.EnFr15KAttrDataset]'s attribute
triples to `fit()` for numbers directly comparable to OpenEA's published
Hits@1=0.338, Hits@10=0.680, MRR=0.451
(``docs/detailed_results_current_approaches_15K.csv``).

Deliberate deviations, confirmed against OpenEA's own reference config
(``run/args/gcnalign_args_15K.json``) and source:

- **Plain SGD, not Adam/Adagrad.** OpenEA's own ``GCN_Align_Unit`` uses
  ``tf.train.GradientDescentOptimizer`` and asserts
  ``args.learning_rate >= 0.01`` -- its own published value is an unusually
  high ``learning_rate=8``, which is intentional fidelity to the reference,
  not a typo. Every other hand-rolled linker in this package uses
  Adam/Adagrad; this one deliberately doesn't, to match what the reference
  actually runs.
- **No mapping matrix.** OpenEA's generic pipeline config labels this
  ``"alignment_module": "mapping"``, but ``gcn_align.py``'s own model code
  never builds or trains a projection matrix -- entities from both KGs share
  one embedding table (disjoint ids, same pattern as
  [RSN4EALinker][linkingtk.algorithms.ea.rsn4ea.RSN4EALinker]), and
  ``source_embedding``/``target_embedding`` are identical passthroughs.
- **Negative sampling refresh cadence is fixed at every 10 epochs**
  (matching OpenEA's ``if i % 10 == 1``), independent of ``eval_every``.
- **Candidate scoring uses Manhattan (L1) distance, not cosine
  similarity.** OpenEA's own config for this method is ``eval_metric:
  "manhattan"``, ``eval_norm: false`` (confirmed by reading
  ``run/args/gcnalign_args_15K.json`` directly) -- and the margin loss
  both branches train under
  ([margin_ranking_loss_l1][linkingtk.algorithms.ea._ea_losses.margin_ranking_loss_l1])
  is itself L1-based, so embeddings are optimized to be *close in L1
  distance*, not high-cosine. Scoring with a metric that matches the
  training geometry, rather than an arbitrary different one, is a real
  correctness fix, not just fidelity to OpenEA's own choice -- confirmed
  empirically on the real EN-FR-15K-V1 benchmark (structural-only):
  Hits@1 0.130 (cosine) -> 0.147 (manhattan, no CSLS). CSLS
  (Cross-domain Similarity Local Scaling, ``csls_k=10`` in OpenEA's own
  config) is a further, separate refinement -- see
  [rank_exhaustive][linkingtk.eval.ranking.rank_exhaustive]'s ``csls_k``
  parameter, used by ``examples/gcn_align_benchmark.py`` but not by
  ``link()`` itself (CSLS needs the full candidate pool's neighbor
  structure, which doesn't fit ``link()``'s post-blocking, per-source
  candidate-list shape).
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial.distance import cdist

from linkingtk.algorithms.base import DEFAULT_BLOCKING, BaseLinker
from linkingtk.algorithms.ea._ea_losses import margin_ranking_loss_l1
from linkingtk.algorithms.ea._gcn_align_torch import (
    build_gcn_align_attr_branch,
    build_gcn_align_branch,
)
from linkingtk.algorithms.ea._gcn_align_training import (
    build_attribute_features,
    build_weighted_adjacency,
    compute_relation_functionality,
    sample_negatives,
)
from linkingtk.algorithms.matching import DEFAULT_MATCHER, Matcher
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity
from linkingtk.core.result import AlignmentResult
from linkingtk.core.source import EntitySource
from linkingtk.exceptions import LinkingTKError, OptionalDependencyError
from linkingtk.utils.device import resolve_device
from linkingtk.utils.graph import Graph, build_id_mappings, map_triples_to_ids, to_triples
from linkingtk.utils.sparse_gcn import coo_to_torch_sparse, normalize_adjacency_coo

if TYPE_CHECKING:
    import numpy.typing as npt

    from linkingtk.utils.graph import Triple

_NEG_REFRESH_INTERVAL = 10


class GCNAlignLinker(BaseLinker):
    """Scores candidate pairs via a 2-layer GCN's propagated embeddings.

    Must be [fit][linkingtk.algorithms.ea.gcn_align.GCNAlignLinker.fit] before
    [link][linkingtk.algorithms.base.BaseLinker.link] can be called. See the
    module docstring for what's ported vs. deliberately deviated from OpenEA,
    and for the structural-only-vs.-``sa`` (structural+attribute) tradeoff.

    Args:
        embedding_dim: Dimensionality of the propagated structural
            embeddings (OpenEA's ``se_dim``). OpenEA's published
            EN-FR-15K-V1 config uses ``100``.
        num_epochs: Training epochs. OpenEA's published value allows up to
            ``2000`` with early stopping.
        learning_rate: Plain SGD's learning rate -- see the module docstring
            for why this is unusually high by this package's conventions.
            OpenEA's published value is ``8``.
        neg_triple_num: Negatives sampled per seed pair per side (``k``),
            refreshed every 10 epochs. OpenEA's published value is ``5``.
        gamma: Margin for
            [margin_ranking_loss_l1][linkingtk.algorithms.ea._ea_losses.margin_ranking_loss_l1].
            OpenEA's published value is ``3``.
        min_weight: Floor applied to every adjacency edge weight -- see
            [build_weighted_adjacency][linkingtk.algorithms.ea._gcn_align_training.build_weighted_adjacency].
            OpenEA's own value (hardcoded, not configurable in the
            reference) is ``0.3``.
        use_attributes: Whether to also train the attribute (``ae``) branch
            -- needs `fit()`'s ``attribute_triples1``/``attribute_triples2``
            to be given. Defaults to ``False`` (structural-only, needs
            nothing beyond plain relational triples).
        attr_dim: Dimensionality of the propagated attribute embeddings
            (OpenEA's ``ae_dim``). Only used if ``use_attributes``. OpenEA's
            published value is ``100``.
        attr_top_fraction: Fraction of the combined attribute-predicate
            vocabulary kept as feature columns -- see
            [build_attribute_features][linkingtk.algorithms.ea._gcn_align_training.build_attribute_features].
            Only used if ``use_attributes``. OpenEA's own value (hardcoded)
            is ``0.7``.
        beta: Weight on the structural half of the concatenated
            ``[se * beta, ae * (1 - beta)]`` scoring embedding. Only used
            if ``use_attributes``. OpenEA's published value is ``0.9``.
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
        learning_rate: float = 8.0,
        neg_triple_num: int = 5,
        gamma: float = 3.0,
        min_weight: float = 0.3,
        use_attributes: bool = False,
        attr_dim: int = 100,
        attr_top_fraction: float = 0.7,
        beta: float = 0.9,
        matching: Matcher = DEFAULT_MATCHER,
        device: str = "cpu",
    ) -> None:
        self.embedding_dim = embedding_dim
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.neg_triple_num = neg_triple_num
        self.gamma = gamma
        self.min_weight = min_weight
        self.use_attributes = use_attributes
        self.attr_dim = attr_dim
        self.attr_top_fraction = attr_top_fraction
        self.beta = beta
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
        attribute_triples1: list[Triple] | None = None,
        attribute_triples2: list[Triple] | None = None,
    ) -> GCNAlignLinker:
        """Propagate structural (and, if enabled, attribute) embeddings through the graph.

        Args:
            dataset1: Source entities. Unused beyond mirroring sibling
                linkers' signatures -- which entities get embeddings is
                determined entirely by ``graph``.
            dataset2: Target entities. See ``dataset1``.
            ground_truth: List of ``(source_id, target_id)`` known-correct
                pairs used as positive pairs in the margin loss.
            graph: The combined relational structure of both KGs (e.g.
                ``to_triples(graph1) + to_triples(graph2)`` from a
                [GraphDatasetLoader][linkingtk.datasets.base.GraphDatasetLoader]'s
                ``load_graphs()``) -- entity ids on both sides must already
                be disjoint, as they are from that loader. Also drives the
                structural adjacency both branches propagate over, even
                when ``use_attributes`` is set.
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
            attribute_triples1: KG1's ``(entity_id, predicate, value)``
                attribute triples, e.g. from
                [EnFr15KAttrDataset.load_attribute_triples][linkingtk.datasets.openea_native._OpenEANativeDataset.load_attribute_triples].
                Required if ``use_attributes`` (set in ``__init__``);
                ignored otherwise.
            attribute_triples2: KG2's own attribute triples. See
                ``attribute_triples1``.

        Returns:
            ``self``, for chaining.

        Raises:
            LinkingTKError: If none of ``ground_truth``'s pairs have both
                ids present in ``graph``'s own triples, if ``use_attributes``
                is set but no attribute triples (or no attribute predicate
                clears ``attr_top_fraction``'s cutoff) are given, or if
                ``device`` is invalid or unavailable.
            OptionalDependencyError: If torch isn't installed.
        """
        try:
            import torch
        except ImportError as exc:
            raise OptionalDependencyError("GCNAlignLinker", "kge") from exc

        device = resolve_device(self.device)
        if random_state is not None:
            torch.manual_seed(random_state)
            torch.cuda.manual_seed_all(random_state)
        rng = np.random.default_rng(random_state)

        triples = to_triples(graph)
        entity_to_id, relation_to_id = build_id_mappings(triples)
        mapped = map_triples_to_ids(triples, entity_to_id, relation_to_id)
        num_entities = len(entity_to_id)

        seed_pairs = [
            (entity_to_id[s], entity_to_id[t])
            for s, t in ground_truth
            if s in entity_to_id and t in entity_to_id
        ]
        if not seed_pairs:
            raise LinkingTKError(
                "None of `ground_truth`'s pairs have both ids present in `graph`'s "
                "own triples; fit() has no seed pairs to train GCN-Align's margin loss with."
            )

        r2f, r2if = compute_relation_functionality(mapped)
        raw_indices, raw_values = build_weighted_adjacency(mapped, r2f, r2if, self.min_weight)
        norm_indices, norm_values = normalize_adjacency_coo(raw_indices, raw_values, num_entities)
        adjacency = coo_to_torch_sparse(
            norm_indices, norm_values, (num_entities, num_entities), device
        )

        model_se = build_gcn_align_branch(num_entities, self.embedding_dim).to(device)
        optimizer_se = torch.optim.SGD(model_se.parameters(), lr=self.learning_rate)

        model_ae: torch.nn.Module | None = None
        optimizer_ae: torch.optim.Optimizer | None = None
        attr_features: torch.Tensor | None = None
        if self.use_attributes:
            if not attribute_triples1 and not attribute_triples2:
                raise LinkingTKError(
                    "GCNAlignLinker(use_attributes=True) needs `attribute_triples1`/"
                    "`attribute_triples2` -- pass entities and attribute triples from "
                    "EnFr15KAttrDataset (or similar), not EnFr15KDataset."
                )
            attr_indices, attr_values, num_attrs = build_attribute_features(
                attribute_triples1 or [],
                attribute_triples2 or [],
                entity_to_id,
                self.attr_top_fraction,
            )
            if num_attrs == 0:
                raise LinkingTKError(
                    "GCNAlignLinker(use_attributes=True): no attribute predicate cleared "
                    "`attr_top_fraction`'s cutoff -- nothing to train the attribute branch with."
                )
            attr_features = coo_to_torch_sparse(
                attr_indices, attr_values, (num_entities, num_attrs), device
            )
            model_ae = build_gcn_align_attr_branch(num_attrs, self.attr_dim).to(device)
            optimizer_ae = torch.optim.SGD(model_ae.parameters(), lr=self.learning_rate)

        val_pairs = [
            (s, t) for s, t in (val_ground_truth or []) if s in entity_to_id and t in entity_to_id
        ]
        best_hits1 = -1.0
        epochs_without_improvement = 0

        pos_left = torch.tensor([s for s, _ in seed_pairs], device=device)
        pos_right = torch.tensor([t for _, t in seed_pairs], device=device)

        def _resample_negatives() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            neg_left_np, neg_right_np, neg2_left_np, neg2_right_np = sample_negatives(
                seed_pairs, num_entities, self.neg_triple_num, rng
            )
            return (
                torch.from_numpy(neg_left_np).long().to(device),
                torch.from_numpy(neg_right_np).long().to(device),
                torch.from_numpy(neg2_left_np).long().to(device),
                torch.from_numpy(neg2_right_np).long().to(device),
            )

        def _combined_embeddings(
            se: npt.NDArray[Any], ae: npt.NDArray[Any] | None
        ) -> npt.NDArray[Any]:
            if ae is None:
                return se
            return np.concatenate([se * self.beta, ae * (1.0 - self.beta)], axis=1)

        neg_left, neg_right, neg2_left, neg2_right = _resample_negatives()

        for epoch in range(self.num_epochs):
            if epoch % _NEG_REFRESH_INTERVAL == 0 and epoch > 0:
                neg_left, neg_right, neg2_left, neg2_right = _resample_negatives()

            embeddings_se = model_se(adjacency)
            loss_se = margin_ranking_loss_l1(
                embeddings_se,
                pos_left,
                pos_right,
                neg_left,
                neg_right,
                neg2_left,
                neg2_right,
                self.gamma,
            )
            optimizer_se.zero_grad()
            loss_se.backward()  # type: ignore[no-untyped-call]
            optimizer_se.step()

            if model_ae is not None and optimizer_ae is not None and attr_features is not None:
                embeddings_ae = model_ae(adjacency, attr_features)
                loss_ae = margin_ranking_loss_l1(
                    embeddings_ae,
                    pos_left,
                    pos_right,
                    neg_left,
                    neg_right,
                    neg2_left,
                    neg2_right,
                    self.gamma,
                )
                optimizer_ae.zero_grad()
                loss_ae.backward()  # type: ignore[no-untyped-call]
                optimizer_ae.step()

            if val_pairs and (epoch + 1) % eval_every == 0:
                with torch.no_grad():
                    se_np = model_se(adjacency).cpu().numpy()
                    ae_np = (
                        model_ae(adjacency, attr_features).cpu().numpy()
                        if model_ae is not None and attr_features is not None
                        else None
                    )
                current_embeds = _combined_embeddings(se_np, ae_np)
                hits1 = _validation_hits1(current_embeds, entity_to_id, val_pairs)
                if hits1 <= best_hits1:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        break
                else:
                    best_hits1 = hits1
                    epochs_without_improvement = 0

        with torch.no_grad():
            final_se = model_se(adjacency).cpu().numpy()
            final_ae = (
                model_ae(adjacency, attr_features).cpu().numpy()
                if model_ae is not None and attr_features is not None
                else None
            )
        final_embeds = _combined_embeddings(final_se, final_ae)
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
            raise LinkingTKError("GCNAlignLinker.link() called before fit().")

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

    Same deliberately-simplified stand-in for OpenEA's own dual-flag
    ``early_stop()`` as
    [MTransELinker][linkingtk.algorithms.ea.mtranse.MTransELinker]'s
    ``_validation_hits1`` -- a plain patience counter over Hits@1, computed
    directly via numpy rather than the full ``Matcher``/``BlockingStrategy``
    pipeline.
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
