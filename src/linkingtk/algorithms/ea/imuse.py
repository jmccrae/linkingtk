"""IMUSE knowledge-graph-embedding linker for Entity Alignment.

He, F., Li, Z., Qiao, Q., Liu, A., & Zhao, L. (2019). Unsupervised Entity
Alignment Using Attribute Triples and Relation Triples. DASFAA 2019.
https://link.springer.com/content/pdf/10.1007%2F978-3-030-18576-3_22.pdf

Like [KDCoELinker][linkingtk.algorithms.ea.kdcoe.KDCoELinker] and
[AttrELinker][linkingtk.algorithms.ea.attre.AttrELinker], this is a
faithful reimplementation of IMUSE's actual training procedure, ported
directly from OpenEA's reference implementation
(https://github.com/nju-websoft/OpenEA/blob/master/src/openea/approaches/imuse.py,
its ``modules/base/initializers.py``/``modules/base/losses.py``, and
``models/basic_model.py``) rather than from a from-scratch reading of the
paper.

**IMUSE is unsupervised** -- unlike every other linker in this family,
[fit][linkingtk.algorithms.ea.imuse.IMUSELinker.fit] takes **no
``ground_truth`` parameter at all**. Reading ``imuse.py`` confirms there's
no seed-pair argument anywhere in OpenEA's own ``IMUSE`` class:

- **Bootstrap alignment from attribute string similarity, before any
  embedding training starts** (ports ``interactive_model``): entity
  attribute predicates are paired across KGs by Levenshtein-ratio
  similarity of their URIs' local names, kept to the 10 pairs with the
  most combined triple support; entities are then paired by the average
  Levenshtein ratio of their values under those aligned predicates. This
  fixed set is IMUSE's *only* source of cross-KG signal.
- **Likely bug found while reading the reference source, not obvious from
  a paper summary**: OpenEA's own ``run_one_ea``/``run_one_ae`` accept a
  candidate match *inside* the inner per-candidate scan loop, on every
  improving match, rather than after scanning all candidates -- so an
  earlier, since-superseded match for the same entity stays in the result
  set and permanently claims its target, even after a strictly better
  match is found later in the same scan. This is inconsistent with the
  textbook-correct version of the identical greedy-accept pattern used a
  few dozen lines away in the same file
  (``get_aligned_attr_pair_by_name_similarity``). Treated as an
  unintentional implementation bug rather than a deliberate design
  choice -- unlike this family's usual "port literal quirks faithfully"
  posture for behavior that's actually load-bearing for a published
  number (e.g. this linker's own full-list epoch re-runs, below) -- this
  port accepts only the single best match found after a full scan. See
  ``_imuse_text.py``'s ``align_entities_by_attribute_values`` docstring.
- **Tractability**: OpenEA brute-forces every candidate entity pair
  (~225M comparisons at this dataset's scale), parallelized across 8
  processes. A single-process port instead indexes attribute values per
  KG and only scores entity pairs that share at least one value (after
  casefolding) under an aligned predicate -- entities with no shared
  aligned-predicate value can only ever score `0` similarity anyway, so
  this loses no *correct* matches from OpenEA's own similarity formula,
  only ones whose only signal is a near-but-not-exact value match. See
  ``_imuse_text.py``'s ``_value_index`` docstring.
- **A real, diagnosed benchmark finding**: at OpenEA's own hardcoded
  ``0.6`` name-similarity threshold, live EN-FR-15K-V1 data lets a
  spurious attribute-predicate pair through --
  ``.../ontology/games``/``foaf/0.1/name`` score `0.667` on local-name
  Levenshtein ratio (genuinely above `0.6`, not a bug), and having more
  combined triple support than the *correct* self-match
  ``foaf/0.1/name``/``foaf/0.1/name``, it wins that predicate's top-10
  slot instead. Verified directly (not assumed): this one wrong predicate
  pair drags the entity-bootstrap step's precision against this dataset's
  own known links down to 54.7%; raising ``name_sim_threshold`` to ``0.9``
  removes it and raises precision to 82.1%, taking live Hits@1 from
  ``0.470`` to ``0.571`` with no other change. This is very likely a
  rehost-specific attribute-predicate-vocabulary difference from OpenEA's
  own original dataset files (not present in whatever they benchmarked
  on), not a bug in this port -- see ``examples/imuse_benchmark.py``,
  which uses ``name_sim_threshold=0.9`` for exactly this reason, while
  the class default here stays at OpenEA's own literal ``0.6``.
- **Structural (SE) half**: once bootstrapping is done, IMUSE trains a
  margin-based TransE embedding over relation triples -- the same shape
  every linker in this family uses, reusing ``_kdcoe_torch.py``'s
  ``KGContext``/``build_kg_context``/``train_structural_epoch`` directly.
  Entity/relation ids come from one **combined** id space across
  both KGs (like [MTransELinker][linkingtk.algorithms.ea.mtranse.MTransELinker]/
  [KDCoELinker][linkingtk.algorithms.ea.kdcoe.KDCoELinker]), *not* the
  shared-id "sharing" merge
  [IPTransELinker][linkingtk.algorithms.ea.iptranse.IPTransELinker]/
  [JAPELinker][linkingtk.algorithms.ea.jape.JAPELinker]/
  [AttrELinker][linkingtk.algorithms.ea.attre.AttrELinker] use -- IMUSE
  never merges rows for aligned pairs; alignment is a separate loss term
  (next point).
- **Alignment loss -- the genuinely new part**: unlike every other linker
  here, IMUSE has **no mapping matrix and no shared-id merge**. Instead,
  a direct ``sum(||ent_embeds[e1] - ent_embeds[e2]||^2)`` loss pulls the
  two (still separate-row) embeddings of each bootstrapped pair together.
  OpenEA's own ``launch_align_training_1epo`` re-runs this over the
  *full* bootstrapped-pair list ``steps`` times per epoch rather than
  sub-batching it -- the same full-list-re-run quirk
  [AttrELinker][linkingtk.algorithms.ea.attre.AttrELinker]'s
  ``train_joint_epoch`` already documents; ported the same way here (see
  ``_imuse_torch.py``'s ``train_align_epoch``).
- **`optimizer="SGD"`** -- like AttrE, not Adagrad (this family's more
  common default).
- **Even though training is unsupervised, early stopping is not**:
  ``val_ground_truth`` is still accepted and used exactly like every other
  linker here (OpenEA's own ``run()`` checks Hits@1 against
  ``kgs.valid_links`` every ``eval_freq`` epochs) -- the "unsupervised"
  claim is specifically about training signal, not about early stopping.

**New dependency**: `rapidfuzz` (added to the ``kge`` optional-dependency
group alongside ``pykeen``/``torch``) for Levenshtein-ratio string
similarity -- OpenEA's own bootstrap relies on ``python-Levenshtein``'s
``.ratio()``; ``rapidfuzz.distance.Indel.normalized_similarity`` computes
the identical formula (``rapidfuzz.distance.Levenshtein`` alone uses a
different substitution cost/normalization -- see ``_imuse_text.py``'s
``levenshtein_ratio`` docstring). Imported lazily (inside functions, not
at module top) in ``_imuse_text.py``, matching this package's established
`torch`-avoidance convention, so importing ``linkingtk.algorithms.ea``
doesn't require the ``kge`` extra just to reach an unrelated linker.

**Same dataset note as JAPE/KDCoE/AttrE**: this uses
[EnFr15KAttrDataset][linkingtk.datasets.openea_native.EnFr15KAttrDataset],
not `EnFr15KDataset`, since IMUSE's entire mechanism depends on attribute
triples and that loader has none.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from linkingtk.algorithms.base import DEFAULT_BLOCKING, BaseLinker
from linkingtk.algorithms.ea._device import resolve_device
from linkingtk.algorithms.ea._imuse_text import bootstrap_alignment
from linkingtk.algorithms.ea._imuse_torch import train_align_epoch
from linkingtk.algorithms.ea._iptranse_torch import validation_hits1
from linkingtk.algorithms.ea._kdcoe_torch import build_kg_context, train_structural_epoch
from linkingtk.algorithms.matching import DEFAULT_MATCHER, Matcher
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity
from linkingtk.core.result import AlignmentResult
from linkingtk.exceptions import LinkingTKError, OptionalDependencyError
from linkingtk.utils.graph import Graph, Triple, build_id_mappings, map_triples_to_ids, to_triples

if TYPE_CHECKING:
    import numpy.typing as npt


class IMUSELinker(BaseLinker):
    """Unsupervised EA: bootstraps its own seed pairs from attribute-value similarity.

    Must be [fit][linkingtk.algorithms.ea.imuse.IMUSELinker.fit] before
    [link][linkingtk.algorithms.base.BaseLinker.link] can be called. Unlike
    every other linker in this family, [fit][linkingtk.algorithms.ea.imuse.IMUSELinker.fit]
    takes no ``ground_truth`` -- see the module docstring.

    Args:
        embedding_dim: Dimensionality of the trained entity/relation
            embeddings. OpenEA's published EN-FR-15K-V1 config uses
            ``100``.
        num_epochs: Training epochs, each pairing one epoch of structural
            triple-loss training with one epoch of alignment-loss
            training. OpenEA's config allows up to ``2000`` with early
            stopping (see ``val_ground_truth`` on
            [fit][linkingtk.algorithms.ea.imuse.IMUSELinker.fit]).
        batch_size: Mini-batch size for structural triple-loss training;
            the alignment-loss step count is derived from it (matching
            OpenEA).
        learning_rate: SGD's learning rate, for both optimizers --
            OpenEA's published value is ``0.01``.
        margin: Hinge margin for the structural triple loss. OpenEA's
            published value is ``1.5``.
        name_sim_threshold: Minimum Levenshtein ratio for two attribute
            predicates' local names to be considered aligned. OpenEA
            hardcodes ``0.6`` at this call site.
        entity_sim_threshold: Minimum average Levenshtein ratio (across
            shared aligned-predicate values) for two entities to be
            considered aligned. OpenEA's published value
            (``sim_thresholds_ent``) is ``0.6``.
        attr_sim_threshold: Same as ``entity_sim_threshold`` but for
            re-aligning attribute predicates from aligned entities
            (``sim_thresholds_attr``, only used if ``bootstrap_iterations
            > 1``). OpenEA's published value is ``0.6``.
        top_k_attribute_pairs: Maximum aligned attribute-predicate pairs
            kept after name-similarity matching, ranked by combined triple
            count. OpenEA hardcodes ``10``.
        bootstrap_iterations: Outer bootstrap rounds (entity alignment,
            then optionally attribute-predicate re-alignment from those
            entities). OpenEA's published default
            (``interactive_model_iter_num``) is ``1``, at which the
            attribute-predicate-realignment step never runs -- see the
            module docstring.
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
        batch_size: int = 5000,
        learning_rate: float = 0.01,
        margin: float = 1.5,
        name_sim_threshold: float = 0.6,
        entity_sim_threshold: float = 0.6,
        attr_sim_threshold: float = 0.6,
        top_k_attribute_pairs: int = 10,
        bootstrap_iterations: int = 1,
        matching: Matcher = DEFAULT_MATCHER,
        device: str = "cpu",
    ) -> None:
        self.embedding_dim = embedding_dim
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.margin = margin
        self.name_sim_threshold = name_sim_threshold
        self.entity_sim_threshold = entity_sim_threshold
        self.attr_sim_threshold = attr_sim_threshold
        self.top_k_attribute_pairs = top_k_attribute_pairs
        self.bootstrap_iterations = bootstrap_iterations
        self.matching = matching
        self.device = device
        self._id_to_vector: dict[str, npt.NDArray[np.floating[Any]]] = {}
        self._fitted = False

    def fit(
        self,
        dataset1: list[Entity],
        dataset2: list[Entity],
        graph: Graph,
        attribute_triples1: list[Triple],
        attribute_triples2: list[Triple],
        random_state: int | None = None,
        val_ground_truth: list[tuple[str, str]] | None = None,
        patience: int = 5,
        eval_every: int = 10,
    ) -> IMUSELinker:
        """Bootstrap an alignment from attribute similarity, then train jointly.

        Args:
            dataset1: Source entities -- used to partition ``graph`` into
                each KG's own triples (same approach
                [KDCoELinker][linkingtk.algorithms.ea.kdcoe.KDCoELinker]
                uses).
            dataset2: Target entities. See ``dataset1``.
            graph: The combined relational structure of both KGs -- entity
                ids on both sides must already be disjoint.
            attribute_triples1: KG1's ``(entity_id, predicate, value)``
                attribute triples -- **required**, not optional (unlike
                [JAPELinker][linkingtk.algorithms.ea.jape.JAPELinker]/
                [KDCoELinker][linkingtk.algorithms.ea.kdcoe.KDCoELinker]):
                IMUSE has no ground truth at all, so bootstrapping from
                attributes is its only source of cross-KG signal, with no
                fallback.
            attribute_triples2: KG2's attribute triples. See
                ``attribute_triples1``.
            random_state: Seed for reproducible training. Left
                unspecified, training is non-deterministic.
            val_ground_truth: Optional held-out pairs used for early
                stopping -- every ``eval_every`` epochs, Hits@1 is checked
                against this set, and training stops after ``patience``
                checks with no improvement. Unlike ``ground_truth``, this
                *is* accepted here -- see the module docstring's note on
                what "unsupervised" does and doesn't cover. If ``None``
                (default), trains the full ``num_epochs`` unconditionally.
            patience: Number of non-improving ``eval_every``-spaced checks
                to tolerate before stopping early. Only used if
                ``val_ground_truth`` is given.
            eval_every: How often (in epochs) to check ``val_ground_truth``.
                Only used if ``val_ground_truth`` is given.

        Returns:
            ``self``, for chaining.

        Raises:
            LinkingTKError: If both ``attribute_triples1`` and
                ``attribute_triples2`` are empty (nothing to bootstrap an
                alignment with), or if bootstrapping finds no confident
                aligned entity pairs at all (nothing to train the
                alignment loss with), or if ``device`` is invalid or
                unavailable.
            OptionalDependencyError: If torch or rapidfuzz isn't installed.
        """
        try:
            import rapidfuzz  # noqa: F401 -- import-only check, used inside _imuse_text.py
            import torch
            import torch.nn.functional as functional
        except ImportError as exc:
            raise OptionalDependencyError("IMUSELinker", "kge") from exc

        if not attribute_triples1 and not attribute_triples2:
            raise LinkingTKError(
                "Both `attribute_triples1` and `attribute_triples2` are empty; "
                "IMUSELinker.fit() has no attribute signal to bootstrap an "
                "alignment with (it has no ground_truth to fall back on)."
            )

        device = resolve_device(self.device)
        if random_state is not None:
            torch.manual_seed(random_state)
            torch.cuda.manual_seed_all(random_state)
        rng = np.random.default_rng(random_state)

        ids1 = {entity.id for entity in dataset1}
        ids2 = {entity.id for entity in dataset2}
        triples = to_triples(graph)
        triples1 = [t for t in triples if t[0] in ids1]
        triples2 = [t for t in triples if t[0] in ids2]

        entity_to_id, relation_to_id = build_id_mappings(triples1 + triples2)
        mapped1 = map_triples_to_ids(triples1, entity_to_id, relation_to_id)
        mapped2 = map_triples_to_ids(triples2, entity_to_id, relation_to_id)
        ctx1 = build_kg_context(mapped1)
        ctx2 = build_kg_context(mapped2)

        bootstrapped_pairs = bootstrap_alignment(
            attribute_triples1,
            attribute_triples2,
            self.name_sim_threshold,
            self.entity_sim_threshold,
            self.attr_sim_threshold,
            self.top_k_attribute_pairs,
            self.bootstrap_iterations,
        )
        seed_pairs = [
            (s, t) for s, t in bootstrapped_pairs if s in entity_to_id and t in entity_to_id
        ]
        if not seed_pairs:
            raise LinkingTKError(
                "Bootstrapping from attribute-value similarity found no confident "
                "aligned entity pairs (or none with both ids present in `graph`'s "
                "own triples); fit() has no seed pairs to train IMUSE's alignment "
                "loss with."
            )
        seed_source_ids = np.array([entity_to_id[s] for s, _ in seed_pairs], dtype=np.int64)
        seed_target_ids = np.array([entity_to_id[t] for _, t in seed_pairs], dtype=np.int64)

        entity_embeds, relation_embeds = self._init_embeddings(
            torch, functional, len(entity_to_id), len(relation_to_id), device
        )
        triple_optimizer = torch.optim.SGD([entity_embeds, relation_embeds], lr=self.learning_rate)
        align_optimizer = torch.optim.SGD([entity_embeds], lr=self.learning_rate)
        align_steps = max(1, math.ceil(len(seed_pairs) / self.batch_size))

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
            train_align_epoch(
                entity_embeds, align_optimizer, seed_source_ids, seed_target_ids, align_steps
            )

            if val_pairs and (epoch + 1) % eval_every == 0:
                with torch.no_grad():
                    current_embeds = functional.normalize(entity_embeds, dim=1).cpu().numpy()
                hits1 = validation_hits1(current_embeds, entity_to_id, val_pairs)
                if hits1 <= best_hits1:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        break
                else:
                    best_hits1 = hits1
                    epochs_without_improvement = 0

        with torch.no_grad():
            final_embeds = functional.normalize(entity_embeds, dim=1).cpu().numpy()
        self._id_to_vector = {
            entity_id: final_embeds[index] for entity_id, index in entity_to_id.items()
        }
        self._fitted = True
        return self

    def _init_embeddings(
        self, torch: Any, functional: Any, num_entities: int, num_relations: int, device: Any
    ) -> tuple[Any, Any]:
        """Truncated-normal init, matching OpenEA's ``init="normal"`` (AttrE's/IPTransE's, too)."""
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
            raise LinkingTKError("IMUSELinker.link() called before fit().")

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
