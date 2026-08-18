"""AttrE knowledge-graph-embedding linker for Entity Alignment.

Trisedya, B. D., Qi, J., & Zhang, R. (2019). Entity Alignment between
Knowledge Graphs Using Attribute Embeddings. AAAI 2019.
https://people.eng.unimelb.edu.au/jianzhongq/papers/AAAI2019_EntityAlignment.pdf

Like [IPTransELinker][linkingtk.algorithms.ea.iptranse.IPTransELinker] and
[JAPELinker][linkingtk.algorithms.ea.jape.JAPELinker], this is a faithful
reimplementation of AttrE's actual training procedure, ported directly
from OpenEA's reference implementation
(https://github.com/nju-websoft/OpenEA/blob/master/src/openea/approaches/attre.py
and its ``modules/train/batch.py`` support code) rather than from a
from-scratch reading of the paper.

**Important finding from reading the reference source, not obvious from
a one-line method summary**: despite AttrE's paper framing it as aligning
entities "using attribute triples only," OpenEA's actual benchmarked
implementation trains a full structural TransE component over relation
triples too, alongside a *separate* character/attribute-embedding (CE)
space -- the two are cross-regularized by a joint cosine-similarity loss.
This is not an optional add-on; it's what OpenEA's own published number
comes from, so this port includes both halves rather than a
structure-free implementation. Two embedding spaces, trained jointly:

- **Structural (SE)**: ``ent_embeds``/``rel_embeds``, plain margin-based
  TransE + negative sampling over relation triples -- reuses
  ``_kdcoe_torch.py``'s ``KGContext``, ``build_kg_context``, and
  ``train_structural_epoch`` directly (already fully generic, no path
  loss, no mapping).
- **Character/attribute (CE)**: ``ent_embeds_ce`` (a *second*,
  independent entity embedding table, sharing the same ids as ``ent_embeds``),
  ``attr_embeds`` (one row per attribute predicate), ``char_embeds`` (one
  row per selected character). Attribute triples are treated as TransE-style
  triples -- ``entity_ce + attribute ≈ compose(value's characters)`` --
  with the value's embedding built compositionally from its characters
  (see ``_attre_torch.py``'s ``compose_value_embeddings``). Negative
  sampling corrupts only the entity endpoint (see ``_attre_text.py``'s
  ``sample_negative_attribute_triples``).

**Alignment module is "sharing"** (``attre_args_15K.json``), identical to
IPTransE's/JAPE's mechanism -- seed pairs get one shared embedding row (in
*both* the SE and CE tables), no mapping matrix. This is the only
cross-KG alignment signal; there is no separate seed-pair loss anywhere
in AttrE's own code. Reuses ``_iptranse_training.py``'s
``build_shared_id_mappings`` directly.

**Joint loss**: ``sum(1 - cos_sim(ent_embeds[e], ent_embeds_ce[e]))`` over
*every* entity in both KGs (not just seed pairs) -- this is what actually
ties the two spaces together, pulling both toward mutual agreement rather
than fixing one side to a projection of the other (contrast
[MTransELinker][linkingtk.algorithms.ea.mtranse.MTransELinker]'s mapping
loss, which only updates its mapping matrix). See ``_attre_torch.py``'s
``train_joint_epoch`` for a note on a curiosity ported literally from
OpenEA's own code (it re-runs this loss over the *entire* entity list
multiple times per epoch, not sub-batched).

No bootstrapping, no co-training loop -- simpler control flow than
[KDCoELinker][linkingtk.algorithms.ea.kdcoe.KDCoELinker]. Per epoch: one
SE epoch, one CE epoch, one joint-loss pass, then periodic validation.

**`optimizer="SGD"`** in OpenEA's published config -- the first method in
this family to use plain SGD rather than Adagrad; ported literally
(``torch.optim.SGD``).

Deviations beyond what ``_attre_text.py``/``_attre_torch.py`` already
document:

- **`attribute_triples1`/`attribute_triples2` are required, not
  optional** -- unlike
  [JAPELinker][linkingtk.algorithms.ea.jape.JAPELinker] (structural-only
  fallback) and [KDCoELinker][linkingtk.algorithms.ea.kdcoe.KDCoELinker]
  (label-fallback always gives *some* description text), AttrE has no
  meaningful degraded mode without attribute triples -- the entire CE half
  and joint loss depend on them, and there's no fallback text source.
  `fit()` raises `LinkingTKError` if both are empty.
- **Final scoring uses the structural (SE) embedding only**: same
  precedent as every other linker in this family that has more than one
  trained representation -- `ent_embeds_ce` regularizes `ent_embeds`
  during training via the joint loss but isn't itself used by `link()`.
- Early stopping uses this repo's own plain patience-counter Hits@1 check
  rather than OpenEA's own two-flag `early_stop()`, and doesn't gate on
  OpenEA's `start_valid` (checks are simply skipped until
  `val_ground_truth` is given and `eval_every` divides the epoch, same
  convention as every other linker in this family).
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from linkingtk.algorithms.base import DEFAULT_BLOCKING, BaseLinker
from linkingtk.algorithms.ea._attre_text import (
    build_value_char_ids,
    clean_attribute_value,
    select_char_vocabulary,
)
from linkingtk.algorithms.ea._attre_torch import train_attr_epoch, train_joint_epoch
from linkingtk.algorithms.ea._iptranse_torch import validation_hits1
from linkingtk.algorithms.ea._iptranse_training import build_shared_id_mappings
from linkingtk.algorithms.ea._kdcoe_torch import build_kg_context, train_structural_epoch
from linkingtk.algorithms.matching import DEFAULT_MATCHER, Matcher
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity
from linkingtk.core.result import AlignmentResult
from linkingtk.exceptions import LinkingTKError, OptionalDependencyError
from linkingtk.utils.device import resolve_device
from linkingtk.utils.graph import Graph, Triple, map_triples_to_ids, to_triples

if TYPE_CHECKING:
    import numpy.typing as npt


class AttrELinker(BaseLinker):
    """Scores candidate pairs via a shared TransE space regularized by character embeddings.

    Must be [fit][linkingtk.algorithms.ea.attre.AttrELinker.fit] before
    [link][linkingtk.algorithms.base.BaseLinker.link] can be called. See
    the module docstring for how this differs from
    [IPTransELinker][linkingtk.algorithms.ea.iptranse.IPTransELinker] and
    [JAPELinker][linkingtk.algorithms.ea.jape.JAPELinker].

    Args:
        embedding_dim: Dimensionality of every trained embedding table
            (entities, relations, character-space entities, attributes,
            characters). OpenEA's published EN-FR-15K-V1 config uses
            ``100``.
        num_epochs: Training epochs, each one pass of structural training,
            one pass of attribute/character training, and one (repeated)
            pass of joint-loss training. OpenEA's config allows up to
            ``2000`` with early stopping (see ``val_ground_truth`` on
            [fit][linkingtk.algorithms.ea.attre.AttrELinker.fit]).
        batch_size: Mini-batch size for both the structural and
            attribute-triple training (each derives its own step count
            from its own triple count), and the divisor for the joint
            loss's repeat count.
        learning_rate: SGD's learning rate, for every optimizer here.
            OpenEA's published value is ``0.01``.
        margin: Hinge margin for both the structural and attribute-triple
            losses. OpenEA's published value is ``1.5``.
        literal_len: Fixed character-sequence length each attribute value
            is truncated/padded to. OpenEA's published value is ``5``.
        char_freq_threshold: Minimum share of total (distinct-value)
            character occurrences required to keep a character in the
            vocabulary; rarer characters map to the padding id. OpenEA's
            published value is ``0.0001``.
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
        literal_len: int = 5,
        char_freq_threshold: float = 0.0001,
        matching: Matcher = DEFAULT_MATCHER,
        device: str = "cpu",
    ) -> None:
        self.embedding_dim = embedding_dim
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.margin = margin
        self.literal_len = literal_len
        self.char_freq_threshold = char_freq_threshold
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
        attribute_triples1: list[Triple],
        attribute_triples2: list[Triple],
        random_state: int | None = None,
        val_ground_truth: list[tuple[str, str]] | None = None,
        patience: int = 5,
        eval_every: int = 10,
    ) -> AttrELinker:
        """Train a shared TransE space jointly with a character/attribute embedding space.

        Args:
            dataset1: Source entities -- also used to partition ``graph``
                into each KG's own triples (same approach
                [IPTransELinker][linkingtk.algorithms.ea.iptranse.IPTransELinker]
                uses).
            dataset2: Target entities. See ``dataset1``.
            ground_truth: List of ``(source_id, target_id)`` known-correct
                pairs. Seeds the shared embedding tables (each pair's
                target is aliased to its source's id, in both the SE and
                CE spaces).
            graph: The combined relational structure of both KGs -- entity
                ids on both sides must already be disjoint.
            attribute_triples1: KG1's ``(entity_id, predicate, value)``
                attribute triples. **Required** -- see the module
                docstring's deviation note.
            attribute_triples2: KG2's attribute triples. **Required**.
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
            LinkingTKError: If ``attribute_triples1`` and
                ``attribute_triples2`` are both empty, or if none of
                ``ground_truth``'s pairs have both ids present in
                ``dataset1``/``dataset2``, or if ``device`` is invalid or
                unavailable.
            OptionalDependencyError: If torch isn't installed.
        """
        if not attribute_triples1 and not attribute_triples2:
            raise LinkingTKError(
                "AttrELinker.fit() needs attribute triples -- `attribute_triples1` "
                "and `attribute_triples2` are both empty, leaving nothing to train "
                "the character/attribute embedding half with."
            )

        try:
            import torch
            import torch.nn.functional as functional
        except ImportError as exc:
            raise OptionalDependencyError("AttrELinker", "kge") from exc

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

        seed_pairs = [(s, t) for s, t in ground_truth if s in ids1 and t in ids2]
        if not seed_pairs:
            raise LinkingTKError(
                "None of `ground_truth`'s pairs have both ids present in "
                "`dataset1`/`dataset2`; fit() has no seed pairs to train "
                "AttrE's shared embedding tables with."
            )

        entity_to_id, relation_to_id = build_shared_id_mappings(
            triples1_labels + triples2_labels, seed_pairs
        )
        mapped1 = map_triples_to_ids(triples1_labels, entity_to_id, relation_to_id)
        mapped2 = map_triples_to_ids(triples2_labels, entity_to_id, relation_to_id)
        se_ctx1 = build_kg_context(mapped1)
        se_ctx2 = build_kg_context(mapped2)

        (
            ce_ctx1,
            ce_ctx2,
            attribute_to_id,
            char_to_id,
            value_char_ids_tensor,
        ) = self._build_attribute_context(
            torch, entity_to_id, attribute_triples1, attribute_triples2, device
        )

        entity_embeds = self._init_embedding(torch, functional, len(entity_to_id), device)
        relation_embeds = self._init_embedding(torch, functional, len(relation_to_id), device)
        entity_embeds_ce = self._init_embedding(torch, functional, len(entity_to_id), device)
        attr_embeds = self._init_embedding(torch, functional, max(1, len(attribute_to_id)), device)
        char_embeds = self._init_embedding(torch, functional, len(char_to_id) + 1, device)

        se_optimizer = torch.optim.SGD([entity_embeds, relation_embeds], lr=self.learning_rate)
        ce_optimizer = torch.optim.SGD(
            [entity_embeds_ce, attr_embeds, char_embeds], lr=self.learning_rate
        )
        joint_optimizer = torch.optim.SGD([entity_embeds, entity_embeds_ce], lr=self.learning_rate)

        joint_entity_ids = np.array(sorted(entity_to_id.values()), dtype=np.int64)
        joint_steps = max(1, math.ceil(len(joint_entity_ids) / self.batch_size))

        val_pairs = [
            (s, t) for s, t in (val_ground_truth or []) if s in entity_to_id and t in entity_to_id
        ]
        best_hits1 = -1.0
        epochs_without_improvement = 0

        for epoch in range(self.num_epochs):
            train_structural_epoch(
                entity_embeds,
                relation_embeds,
                se_optimizer,
                se_ctx1,
                se_ctx2,
                rng,
                self.batch_size,
                self.margin,
            )
            train_attr_epoch(
                entity_embeds_ce,
                attr_embeds,
                char_embeds,
                value_char_ids_tensor,
                ce_optimizer,
                ce_ctx1,
                ce_ctx2,
                rng,
                self.batch_size,
                self.margin,
            )
            train_joint_epoch(
                entity_embeds, entity_embeds_ce, joint_optimizer, joint_entity_ids, joint_steps
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

    def _build_attribute_context(
        self,
        torch: Any,
        entity_to_id: dict[str, int],
        attribute_triples1: list[Triple],
        attribute_triples2: list[Triple],
        device: Any,
    ) -> tuple[Any, Any, dict[str, int], dict[str, int], Any]:
        """Clean values, build the character/attribute/value id spaces, map triples to ids."""

        def clean(triples: list[Triple]) -> list[Triple]:
            return [
                (entity_id, predicate, clean_attribute_value(value))
                for entity_id, predicate, value in triples
                if entity_id in entity_to_id
            ]

        cleaned1 = clean(attribute_triples1)
        cleaned2 = clean(attribute_triples2)
        all_triples = cleaned1 + cleaned2

        attribute_to_id = {
            predicate: index
            for index, predicate in enumerate(sorted({p for _, p, _ in all_triples}))
        }
        all_values = [value for _, _, value in all_triples]
        char_to_id = select_char_vocabulary(all_values, self.char_freq_threshold)
        value_char_ids_by_value = build_value_char_ids(all_values, char_to_id, self.literal_len)
        value_to_id = {value: index for index, value in enumerate(sorted(value_char_ids_by_value))}
        if value_to_id:
            value_char_ids_array = np.stack(
                [value_char_ids_by_value[value] for value in sorted(value_char_ids_by_value)]
            )
        else:
            value_char_ids_array = np.empty((0, self.literal_len), dtype=np.int64)
        value_char_ids_tensor = torch.from_numpy(value_char_ids_array).long().to(device)

        def map_triples(triples: list[Triple]) -> npt.NDArray[np.int64]:
            rows = [(entity_to_id[e], attribute_to_id[a], value_to_id[v]) for e, a, v in triples]
            return np.array(rows, dtype=np.int64) if rows else np.empty((0, 3), dtype=np.int64)

        def attribute_kg_context(mapped: npt.NDArray[np.int64]) -> Any:
            # build_kg_context's entity_pool is `union(column 0, column 2)`, correct for
            # (head, relation, tail) relation triples where both endpoints are entities --
            # wrong here, since column 2 is a *value* id, not an entity id, from a
            # completely different id space. Replace it with the entity-only pool
            # (column 0 alone) before this context is used for negative sampling.
            ctx = build_kg_context(mapped)
            entity_pool = np.unique(mapped[:, 0]) if len(mapped) else np.empty(0, dtype=np.int64)
            return ctx._replace(entity_pool=entity_pool)

        ce_ctx1 = attribute_kg_context(map_triples(cleaned1))
        ce_ctx2 = attribute_kg_context(map_triples(cleaned2))
        return ce_ctx1, ce_ctx2, attribute_to_id, char_to_id, value_char_ids_tensor

    def _init_embedding(self, torch: Any, functional: Any, size: int, device: Any) -> Any:
        """Truncated-normal init, matching OpenEA's ``init="normal"`` (same as IPTransE's)."""
        std = 1.0 / math.sqrt(self.embedding_dim)
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

    def link(
        self,
        dataset1: list[Entity],
        dataset2: list[Entity],
        graph: Graph = None,
        blocking: BlockingStrategy = DEFAULT_BLOCKING,
    ) -> list[AlignmentResult]:
        if not self._fitted:
            raise LinkingTKError("AttrELinker.link() called before fit().")

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
