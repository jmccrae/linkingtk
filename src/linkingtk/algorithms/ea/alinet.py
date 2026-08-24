"""AliNet knowledge-graph-embedding linker for Entity Alignment.

Sun, Z., Wang, C., Hu, W., Chen, M., Dai, J., Zhang, W., & Qu, Y. (2020).
Knowledge Graph Alignment Network with Gated Multi-hop Neighborhood
Aggregation. AAAI 2020. https://ojs.aaai.org/index.php/AAAI/article/view/5354

The third GNN-based EA linker in this package (see #18's parent issue,
#42's [GCNAlignLinker][linkingtk.algorithms.ea.gcn_align.GCNAlignLinker],
#43's [RDGCNLinker][linkingtk.algorithms.ea.rdgcn.RDGCNLinker]). AliNet
combines a 1-hop [GraphConvolution][linkingtk.algorithms.ea._alinet_torch.build_alinet_model]
branch with a 2-hop gated-attention branch at every layer but the last,
fusing them with a highway gate, and represents each entity by
concatenating every layer's (plus the initial embedding's) output --
letting non-isomorphic local neighborhoods between the two KGs still align
through a longer-range, multi-hop signal. This is a faithful port of
OpenEA's reference implementation
(https://github.com/nju-websoft/OpenEA/blob/master/src/openea/approaches/alinet.py),
not a from-scratch reading of the paper -- see
[_alinet_training][linkingtk.algorithms.ea._alinet_training] and
[_alinet_torch][linkingtk.algorithms.ea._alinet_torch] for the algorithm's
own docstrings, and this module's deviations below (confirmed against
OpenEA's own reference config, ``run/args/alinet_args_15K.json``, and
source):

- **Bootstrapping is implemented but off by default, matching what
  OpenEA's own published config actually runs -- not the milestone
  planning assumption that it's "core to published performance."**
  Reading ``AliNet.run()`` directly: bootstrapping
  (``augment_neighborhood()``) is only called ``if self.args.sim_th >
  0.0``, and the real published config's own value is ``"sim_th": 0.0`` --
  ``0.0 > 0.0`` is `False`, so **OpenEA's own published EN-FR-15K-V1
  numbers never invoke bootstrapping at all**. This port replicates that
  exactly: `sim_th` defaults to `0.0` (bootstrapping off); pass
  `sim_th > 0.0` to opt in.
- **No published EN-FR-15K-V1 number exists for AliNet to benchmark
  against** in ``docs/detailed_results_current_approaches_15K.csv`` --
  added to OpenEA after that CSV's last update; its numbers live only in
  an external Google Sheet linked from OpenEA's README, not reliably
  fetchable. `examples/alinet_benchmark.py` exists for manual
  sanity-checking only, with no target to compare against.
- **The relation-consistency auxiliary loss (`compute_rel_loss`,
  ``rel_param``) is not implemented.** OpenEA's own published config uses
  a small non-zero ``rel_param: 0.01`` (i.e. it *is* nominally active,
  correcting an earlier assumption that it's off by default) -- skipped
  here since no numeric target exists to validate it against anyway, and
  it needs its own ``rel_win_size``-batched relation-window sampling
  machinery for what's a secondary loss term.
- **Uniform negative sampling only.** OpenEA's own published config uses
  ``neg_sampling: "truncated"`` (nearest-neighbor-restricted negatives
  after the first validation checkpoint) -- not implemented; see
  [sample_uniform_cross_kg_negatives][linkingtk.algorithms.ea._alinet_training.sample_uniform_cross_kg_negatives]'s
  docstring.
- **Dropout and L2 weight regularization are not implemented** -- both
  faithful no-ops (OpenEA's own config uses `dropout: 0.0`, and its
  training graph never sums the declared weight regularizers into the
  actual loss). See
  [_alinet_torch][linkingtk.algorithms.ea._alinet_torch]'s module
  docstring.
- **Candidate scoring uses cosine similarity, unlike
  [GCNAlignLinker][linkingtk.algorithms.ea.gcn_align.GCNAlignLinker]/
  [RDGCNLinker][linkingtk.algorithms.ea.rdgcn.RDGCNLinker] (both switched
  to Manhattan distance -- see their own module docstrings for why).**
  Checked, not overlooked: OpenEA's own config for this method is
  ``eval_metric: "inner"`` (raw dot product) over embeddings its own
  ``_eval_test_embeddings``/``save`` L2-normalize immediately before
  scoring -- and
  [AliNetModel][linkingtk.algorithms.ea._alinet_torch.build_alinet_model]'s
  own ``forward()`` already returns L2-normalized output (the final
  ``functional.normalize(concatenated, dim=1)`` call), so raw inner
  product over these embeddings is mathematically identical to cosine
  similarity (``a . b == cosine(a, b)`` when ``|a| = |b| = 1``). No
  change needed here.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from linkingtk.algorithms.base import DEFAULT_BLOCKING, BaseLinker
from linkingtk.algorithms.ea._alinet_torch import build_alinet_model
from linkingtk.algorithms.ea._alinet_training import (
    enhance_triples,
    generate_2hop_pairs,
    pairs_to_symmetric_adjacency,
    run_bootstrapping_round,
    sample_uniform_cross_kg_negatives,
)
from linkingtk.algorithms.ea._ea_losses import margin_ranking_loss_l2_squared
from linkingtk.algorithms.ea._rdgcn_training import build_primal_adjacency
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
    import torch


class AliNetLinker(BaseLinker):
    """Scores candidate pairs via AliNet's gated multi-hop GCN/attention embeddings.

    Must be [fit][linkingtk.algorithms.ea.alinet.AliNetLinker.fit] before
    [link][linkingtk.algorithms.base.BaseLinker.link] can be called. See
    the module docstring for what's ported vs. deliberately deviated from
    OpenEA.

    Args:
        layer_dims: Layer width sequence -- ``layer_dims[0]`` is the
            initial embedding's dimensionality, each subsequent entry one
            more GCN+attention+highway layer's output width. OpenEA's
            published EN-FR-15K-V1 config uses ``[500, 400, 300]``.
        num_epochs: Training epochs. OpenEA's config allows up to ``2000``
            with early stopping.
        learning_rate: Adam's learning rate. OpenEA's published value is
            ``0.001``.
        batch_size: Positive pairs sampled per training step (with
            replacement, matching OpenEA). OpenEA's published value is
            ``3000``.
        neg_triple_num: Negatives drawn per positive pair (``k``, both
            sides independently). OpenEA's published value is ``10``.
        neg_margin: Margin for
            [margin_ranking_loss_l2_squared][linkingtk.algorithms.ea._ea_losses.margin_ranking_loss_l2_squared].
            OpenEA's published value is ``1.5``.
        neg_margin_balance: Weight applied to the negative-pair loss term.
            OpenEA's published value is ``0.1``.
        sim_th: Minimum similarity for a bootstrapped pair to be accepted
            -- **also the bootstrapping on/off switch**: OpenEA's own
            ``run()`` only calls its bootstrapping step if ``sim_th >
            0.0``, and its own published config uses ``0.0`` (off). See
            the module docstring.
        bootstrap_start_epoch: First epoch a bootstrapping round may run,
            if `sim_th > 0`. OpenEA's published value (``start_augment *
            eval_freq``) is ``2 * 10 = 20``.
        matching: Strategy used to resolve scored candidates into final
            links. Defaults to
            [GreedyMatcher][linkingtk.algorithms.matching.GreedyMatcher].
        device: Torch device to train on, e.g. ``"cpu"`` (default) or
            ``"cuda"``/``"cuda:0"``. Trained embeddings are always stored
            as CPU numpy arrays regardless of this setting.
    """

    def __init__(
        self,
        layer_dims: list[int] | None = None,
        num_epochs: int = 500,
        learning_rate: float = 0.001,
        batch_size: int = 3000,
        neg_triple_num: int = 10,
        neg_margin: float = 1.5,
        neg_margin_balance: float = 0.1,
        sim_th: float = 0.0,
        bootstrap_start_epoch: int = 20,
        matching: Matcher = DEFAULT_MATCHER,
        device: str = "cpu",
    ) -> None:
        self.layer_dims = layer_dims if layer_dims is not None else [500, 400, 300]
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.neg_triple_num = neg_triple_num
        self.neg_margin = neg_margin
        self.neg_margin_balance = neg_margin_balance
        self.sim_th = sim_th
        self.bootstrap_start_epoch = bootstrap_start_epoch
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
    ) -> AliNetLinker:
        """Train the gated multi-hop GCN/attention stack, optionally bootstrapping new pairs.

        Args:
            dataset1: Source entities -- also used to partition ``graph``
                into each KG's own triples (same approach
                [IPTransELinker][linkingtk.algorithms.ea.iptranse.IPTransELinker]
                uses), needed for
                [enhance_triples][linkingtk.algorithms.ea._alinet_training.enhance_triples]/
                [generate_2hop_pairs][linkingtk.algorithms.ea._alinet_training.generate_2hop_pairs].
            dataset2: Target entities. See ``dataset1``.
            ground_truth: List of ``(source_id, target_id)`` known-correct
                pairs used as positive pairs in the margin loss, and as
                the initial bootstrapping seed set.
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
                checks with no improvement. Also the cadence bootstrapping
                rounds run on, when enabled. If ``None`` (default), trains
                the full ``num_epochs`` unconditionally (bootstrapping
                still runs on the ``eval_every`` cadence if enabled).
            patience: Number of non-improving ``eval_every``-spaced checks
                to tolerate before stopping early. Only used if
                ``val_ground_truth`` is given.
            eval_every: How often (in epochs) to check ``val_ground_truth``
                and (if enabled) run a bootstrapping round.

        Returns:
            ``self``, for chaining.

        Raises:
            LinkingTKError: If none of ``ground_truth``'s pairs have both
                ids present in ``graph``'s own triples, or if ``device`` is
                invalid or unavailable.
            OptionalDependencyError: If torch isn't installed.
        """
        try:
            import torch
        except ImportError as exc:
            raise OptionalDependencyError("AliNetLinker", "kge") from exc

        device = resolve_device(self.device)
        if random_state is not None:
            torch.manual_seed(random_state)
            torch.cuda.manual_seed_all(random_state)
        rng = np.random.default_rng(random_state)

        ids1 = {entity.id for entity in dataset1}
        ids2 = {entity.id for entity in dataset2}
        triples = to_triples(graph)
        entity_to_id, relation_to_id = build_id_mappings(triples)
        mapped = map_triples_to_ids(triples, entity_to_id, relation_to_id)
        num_entities = len(entity_to_id)

        triples1 = [t for t in triples if t[0] in ids1]
        triples2 = [t for t in triples if t[0] in ids2]
        mapped1 = map_triples_to_ids(triples1, entity_to_id, relation_to_id)
        mapped2 = map_triples_to_ids(triples2, entity_to_id, relation_to_id)

        seed_pairs = [
            (entity_to_id[s], entity_to_id[t])
            for s, t in ground_truth
            if s in entity_to_id and t in entity_to_id
        ]
        if not seed_pairs:
            raise LinkingTKError(
                "None of `ground_truth`'s pairs have both ids present in `graph`'s "
                "own triples; fit() has no seed pairs to train AliNet's margin loss with."
            )

        entities1_ids = [entity_to_id[eid] for eid in ids1 if eid in entity_to_id]
        entities2_ids = [entity_to_id[eid] for eid in ids2 if eid in entity_to_id]

        two_hop_pairs = generate_2hop_pairs(mapped1) | generate_2hop_pairs(mapped2)
        two_indices, two_values = pairs_to_symmetric_adjacency(two_hop_pairs, num_entities)
        two_indices, two_values = normalize_adjacency_coo(two_indices, two_values, num_entities)
        two_hop_adjacency = coo_to_torch_sparse(
            two_indices, two_values, (num_entities, num_entities), device
        )

        def _build_one_hop_adjacency(pairs: list[tuple[int, int]]) -> torch.Tensor:
            enhanced1, enhanced2 = enhance_triples(mapped1, mapped2, pairs)
            all_triples = np.array(
                mapped.tolist() + list(enhanced1) + list(enhanced2), dtype=np.int64
            )
            indices, values = build_primal_adjacency(all_triples, num_entities)
            indices, values = normalize_adjacency_coo(indices, values, num_entities)
            return coo_to_torch_sparse(indices, values, (num_entities, num_entities), device)

        model = build_alinet_model(num_entities, self.layer_dims).to(device)
        _set_adjacency(model, _build_one_hop_adjacency(seed_pairs), two_hop_adjacency)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        aligned1 = {source for source, _target in seed_pairs}
        aligned2 = {target for _source, target in seed_pairs}
        unaligned1 = [entity for entity in entities1_ids if entity not in aligned1]
        unaligned2 = [entity for entity in entities2_ids if entity not in aligned2]

        val_pairs = [
            (s, t) for s, t in (val_ground_truth or []) if s in entity_to_id and t in entity_to_id
        ]
        best_hits1 = -1.0
        epochs_without_improvement = 0

        for epoch in range(self.num_epochs):
            model.train()
            batch_size = min(self.batch_size, len(seed_pairs))
            batch_idx = rng.integers(0, len(seed_pairs), size=batch_size)
            pos_left = torch.tensor([seed_pairs[i][0] for i in batch_idx], device=device)
            pos_right = torch.tensor([seed_pairs[i][1] for i in batch_idx], device=device)

            neg_left_np, neg_right_np = sample_uniform_cross_kg_negatives(
                entities1_ids,
                entities2_ids,
                batch_size,
                self.neg_triple_num,
                rng,
                exclude_pairs=set(seed_pairs),
            )
            neg_left = torch.from_numpy(neg_left_np).long().to(device)
            neg_right = torch.from_numpy(neg_right_np).long().to(device)

            embeddings = model()
            loss = margin_ranking_loss_l2_squared(
                embeddings,
                pos_left,
                pos_right,
                neg_left,
                neg_right,
                self.neg_margin,
                self.neg_margin_balance,
            )
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

            if (epoch + 1) % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    current_embeds = model().cpu().numpy()
                model.train()

                if val_pairs:
                    hits1 = _validation_hits1(current_embeds, entity_to_id, val_pairs)
                    if hits1 <= best_hits1:
                        epochs_without_improvement += 1
                        if epochs_without_improvement >= patience:
                            break
                    else:
                        best_hits1 = hits1
                        epochs_without_improvement = 0

                if self.sim_th > 0.0 and epoch + 1 >= self.bootstrap_start_epoch:
                    new_seed_pairs, unaligned1, unaligned2 = run_bootstrapping_round(
                        current_embeds, seed_pairs, unaligned1, unaligned2, self.sim_th
                    )
                    if len(new_seed_pairs) > len(seed_pairs):
                        seed_pairs = new_seed_pairs
                        _set_adjacency(
                            model, _build_one_hop_adjacency(seed_pairs), two_hop_adjacency
                        )

        model.eval()
        with torch.no_grad():
            final_embeds = model().cpu().numpy()
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
            raise LinkingTKError("AliNetLinker.link() called before fit().")

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


def _set_adjacency(
    model: torch.nn.Module, one_hop_adjacency: torch.Tensor, two_hop_adjacency: torch.Tensor
) -> None:
    """``model.set_adjacency(...)``, typed past a PyTorch stub gap.

    ``torch.nn.Module.__getattr__``'s stub return type is ``Tensor |
    Module`` for any dynamically-registered attribute; mypy can't see
    through [build_alinet_model][linkingtk.algorithms.ea._alinet_torch.build_alinet_model]'s
    ``-> torch.nn.Module`` factory return type to know ``set_adjacency``
    is concretely a bound method -- same precedent as
    ``rsn4ea.py``'s ``_entity_embedding_weight``.
    """
    cast(Any, model).set_adjacency(one_hop_adjacency, two_hop_adjacency)


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
    similarities = cosine_similarity(source_matrix, target_matrix)
    predicted = np.argmax(similarities, axis=1)
    correct = sum(1 for i, j in enumerate(predicted) if j == i)
    return correct / len(val_pairs)
