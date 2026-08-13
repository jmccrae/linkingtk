"""MultiKE knowledge-graph-embedding linker for Entity Alignment.

Zhang, Q., Sun, Z., Hu, W., Chen, M., Guo, L., & Qu, Y. (2019). Multi-view
Knowledge Graph Embedding for Entity Alignment. IJCAI 2019.
https://www.ijcai.org/proceedings/2019/0754.pdf

By far the most complex method in this family (see
[MTransELinker][linkingtk.algorithms.ea.mtranse.MTransELinker],
[IPTransELinker][linkingtk.algorithms.ea.iptranse.IPTransELinker],
[JAPELinker][linkingtk.algorithms.ea.jape.JAPELinker],
[KDCoELinker][linkingtk.algorithms.ea.kdcoe.KDCoELinker],
[AttrELinker][linkingtk.algorithms.ea.attre.AttrELinker],
[IMUSELinker][linkingtk.algorithms.ea.imuse.IMUSELinker]) -- a faithful
reimplementation of MultiKE's actual training procedure, ported directly
from OpenEA's reference implementation
(https://github.com/nju-websoft/OpenEA/blob/master/src/openea/approaches/multi_ke.py,
931 lines, read in full, plus ``literal_encoder.py``,
``predicate_alignmnet.py``, ``modules/base/losses.py``,
``modules/load/read.py``, ``models/basic_model.py``) rather than a
from-scratch paper reading.

**MultiKE is supervised** (unlike IMUSE): seed pairs are its only real
supervision channel, via literal entity-id substitution into each KG's own
triples (see ``_multike_text.generate_cross_kg_relation_triples``/
``generate_cross_kg_attribute_triples``, ported from ``read.py``'s
``generate_sup_relation_triples``/``generate_sup_attribute_triples``).

## Architecture: three views unified into one shared table

1. **Name view** (``name_embeds``): a literal-text embedding of each
   entity's local name, computed once and **frozen** for the rest of
   training. See "The literal encoder" below for the one place this port
   genuinely departs from a faithful reimplementation.
2. **Relation view** (``rv_ent_embeds``/``rel_embeds``): logistic-loss
   TransE over relation triples (``softplus(||h+r-t||^2) +
   softplus(-||h'+r-t'||^2)``) -- **not** the margin-based loss every
   other method in this family uses; MultiKE hardcodes this, ignoring the
   reference's own configurable ``args.loss``.
3. **Attribute view** (``av_ent_embeds``/``attr_embeds``): a small CNN
   (``_multike_torch.build_attribute_scorer``) predicts a "head"
   representation from ``(attribute, value)`` pairs, scored via negative
   squared distance against the real head.
4. **Shared space** (``ent_embeds``, used for every eval call --
   confirmed in ``basic_model.py``'s ``valid()``/``test()``, both always
   read ``self.ent_embeds`` regardless of which views exist): trained by
   alignment terms baked directly into the relation/attribute view losses
   *and* a separate common-space-learning step
   (``_multike_torch.train_common_space_epoch``) pulling ``ent_embeds``
   toward all three views for *every* entity, at its own learning rate
   (``common_space_learning_rate``, distinct from the main
   ``learning_rate``).
5. **Cross-KG entity inference** (the actual supervision channel): for
   each seed pair, every KG1 triple touching the source entity is
   relabeled with the target's id and trained back into the shared
   tables -- see ``_multike_text.py``'s substitution functions.
6. **Predicate soft-alignment**: relation and attribute predicates are
   aligned across KGs by mutual-best-match Levenshtein ratio on local
   names (``_multike_text.align_predicates_by_name``), feeding a
   per-triple weight in the attribute view's main loss and two more
   cross-KG inference loss terms, gated to start at
   ``start_predicate_soft_alignment``.

## Three deliberate scope cuts, documented plainly

- **Space mapping** (``nv_mapping``/``rv_mapping``/``av_mapping``
  orthogonal matrices, ``train_shared_space_mapping_1epo``): defined in
  the reference's ``init()`` but the training loop that calls it is
  **commented out in its own `run()`** (``multi_ke.py:923-926``) --
  genuinely dead code in the reference's own published run, confirmed by
  reading `run()` itself. Not ported at all.
- **Periodic predicate-alignment re-estimation**
  (``update_predicate_alignment``/``find_predicate_alignment_by_embedding``,
  every 10 epochs): refines the Levenshtein-based predicate pairs using
  the training embeddings' own cosine similarity. This port computes the
  Levenshtein-based alignment once at ``fit()`` start and keeps it fixed
  for the whole run (mirrors IMUSE's own fixed-bootstrap precedent) -- a
  refinement pass over an already-working initial signal, not the primary
  mechanism.
- **Truncated/nearest-neighbor negative sampling**: not ported, matching
  every other linker in this family's established uniform-negative-sampling
  precedent (not a new deviation specific to MultiKE).

Also confirmed by reading every loss graph in full (not obvious from a
method-name skim, see ``_multike_torch.py``'s module docstring for the
full breakdown): **only the main relation-view loss ever uses negative
sampling** -- every other loss (attribute view, all four cross-KG
inference terms) is purely positive/regression-style. And
``PredicateAlignModel.relation_triples_w_weights`` is computed by the
reference but never actually consumed anywhere in its own training
methods -- not ported, since it's unused in the reference itself.

## The literal encoder

OpenEA's own ``name_embeds``/``literal_embeds`` come from a pretrained
**English-only** word2vec/fastText file plus a custom stacked autoencoder
trained via reconstruction loss to compress bag-of-word-vectors down to
``embedding_dim``. This port instead uses a small pretrained
**multilingual** transformer (``transformers`` -- already a hard
dependency of this repo, zero new install) via
``_multike_literal.encode_literals``: mean-pooled over the attention
mask, projected down to ``embedding_dim`` via a **fixed random
orthogonal projection** (no training loop needed -- the transformer
already produces well-formed semantic structure). Real multilingual
semantic signal for cross-lingual name matching (e.g.
"United States"/"États-Unis"), which an English-only fastText file
(``_kdcoe_text.py``'s existing ``load_fasttext_vectors`` infrastructure
was considered and is a real precedent in this repo, but doesn't fit a
cross-lingual signal) or a from-scratch in-corpus word2vec couldn't
provide. Default model: ``distilbert-base-multilingual-cased``. First run
downloads it from the Hub (~540MB), cached afterward.

## Other implementation notes

- **Combined, not shared-id, entity space**: like
  [MTransELinker][linkingtk.algorithms.ea.mtranse.MTransELinker]/
  [KDCoELinker][linkingtk.algorithms.ea.kdcoe.KDCoELinker]/
  [IMUSELinker][linkingtk.algorithms.ea.imuse.IMUSELinker] -- MultiKE
  never merges seed pairs into one row; alignment is entirely loss-driven.
- **Which embedding tables get L2-normalized on every read isn't
  uniform** -- ``rv_ent_embeds``/``rel_embeds``/``av_ent_embeds``/
  ``ent_embeds`` always are; ``attr_embeds`` never is (ported from the
  reference's own ``xavier_init(..., is_l2_norm=False)`` for that one
  table); see ``_multike_torch.py``'s module docstring for the full
  detail.
- **`optimizer="SGD"`**, matching AttrE/IMUSE's precedent in this family.
- **Final scoring uses the shared ``ent_embeds`` directly**, cosine
  similarity, no mapping -- matches ``basic_model.py``'s ``valid()``/
  ``test()`` always reading ``self.ent_embeds``.
- **`attribute_triples1`/`attribute_triples2` are required** (raises
  `LinkingTKError` if both empty) -- the attribute view, the literal
  encoder's value vocabulary, and the predicate soft-alignment mechanism
  all depend on them, same posture as AttrE.
- Early stopping uses this repo's own plain patience-counter Hits@1 check
  rather than OpenEA's own two-flag ``early_stop()``, and doesn't gate on
  OpenEA's ``start_valid``, matching every other linker in this family.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from linkingtk.algorithms.base import DEFAULT_BLOCKING, BaseLinker
from linkingtk.algorithms.ea._iptranse_torch import validation_hits1
from linkingtk.algorithms.ea._kdcoe_torch import build_kg_context
from linkingtk.algorithms.ea._multike_literal import encode_literals
from linkingtk.algorithms.ea._multike_text import (
    align_predicates_by_name,
    clean_attribute_value,
    filter_frequent_predicates,
    generate_cross_kg_attribute_triples,
    generate_cross_kg_relation_triples,
    substitute_predicate_triples,
    weight_attribute_triples,
)
from linkingtk.algorithms.ea._multike_torch import (
    build_attribute_context,
    build_attribute_scorer,
    train_attribute_view_epoch,
    train_common_space_epoch,
    train_cross_kg_attribute_entity_epoch,
    train_cross_kg_attribute_predicate_epoch,
    train_cross_kg_relation_entity_epoch,
    train_cross_kg_relation_predicate_epoch,
    train_relation_view_epoch,
)
from linkingtk.algorithms.matching import DEFAULT_MATCHER, Matcher
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity, label_texts
from linkingtk.core.result import AlignmentResult
from linkingtk.datasets._util import label_from_raw
from linkingtk.exceptions import LinkingTKError, OptionalDependencyError
from linkingtk.utils.graph import Graph, Triple, build_id_mappings, map_triples_to_ids, to_triples

if TYPE_CHECKING:
    import numpy.typing as npt


class MultiKELinker(BaseLinker):
    """Scores candidate pairs via a shared embedding table unifying name/relation/attribute views.

    Must be [fit][linkingtk.algorithms.ea.multike.MultiKELinker.fit]
    before [link][linkingtk.algorithms.base.BaseLinker.link] can be
    called. See the module docstring for the full architecture and the
    three documented scope cuts.

    Args:
        embedding_dim: Dimensionality of every trained embedding table.
            OpenEA's published EN-FR-15K-V1 config uses ``100``.
        num_epochs: Training epochs, each one pass of relation-view,
            common-space, cross-KG relation-entity (and, once warmed up,
            cross-KG relation-predicate) training, then the same four
            steps for the attribute view. OpenEA's config allows up to
            ``2000`` with early stopping (see ``val_ground_truth`` on
            [fit][linkingtk.algorithms.ea.multike.MultiKELinker.fit]).
        batch_size: Mini-batch size for the relation view's main loss and
            its two cross-KG inference terms.
        entity_batch_size: Mini-batch size for common-space learning.
        attribute_batch_size: Mini-batch size for the attribute view's
            main loss and its two cross-KG inference terms.
        learning_rate: SGD's learning rate for every optimizer except
            common-space learning. OpenEA's published value is ``0.001``.
        common_space_learning_rate: SGD's learning rate for common-space
            learning specifically (OpenEA's own ``ITC_learning_rate``,
            deliberately distinct from ``learning_rate``). Published
            value is ``0.004``.
        neg_triple_num: Negatives sampled per positive relation triple --
            only the main relation-view loss uses negative sampling at
            all (see the module docstring). Published value is ``10``.
        cv_weight: Weight applied to the common-space-learning loss.
            Published value is ``1.0``.
        predicate_init_sim: Levenshtein-ratio threshold (of humanized
            local names) for the one-time mutual-best-match predicate
            alignment. Published value is ``0.9``.
        predicate_soft_sim: Similarity floor used to rescale an aligned
            predicate pair's weight (``_multike_text.zoom_weight``).
            Published value is ``0.80``.
        start_predicate_soft_alignment: Epoch (0-indexed) at which the two
            predicate-alignment cross-KG loss terms start contributing --
            a warm-up gate. Published value is ``10``.
        min_predicate_triple_count: Attribute predicates with fewer than
            this many combined (both-KG) triples are dropped before
            anything else. Published value is ``10``.
        literal_encoder_model: Hugging Face model id for the literal
            encoder (see the module docstring's "The literal encoder"
            section). Defaults to ``distilbert-base-multilingual-cased``.
        literal_max_length: Token-truncation length passed to the literal
            encoder for every name/value string.
        matching: Strategy used to resolve scored candidates into final
            links. Defaults to
            [GreedyMatcher][linkingtk.algorithms.matching.GreedyMatcher].
    """

    def __init__(
        self,
        embedding_dim: int = 100,
        num_epochs: int = 2000,
        batch_size: int = 5000,
        entity_batch_size: int = 5000,
        attribute_batch_size: int = 5000,
        learning_rate: float = 0.001,
        common_space_learning_rate: float = 0.004,
        neg_triple_num: int = 10,
        cv_weight: float = 1.0,
        predicate_init_sim: float = 0.9,
        predicate_soft_sim: float = 0.80,
        start_predicate_soft_alignment: int = 10,
        min_predicate_triple_count: int = 10,
        literal_encoder_model: str = "distilbert-base-multilingual-cased",
        literal_max_length: int = 16,
        matching: Matcher = DEFAULT_MATCHER,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.entity_batch_size = entity_batch_size
        self.attribute_batch_size = attribute_batch_size
        self.learning_rate = learning_rate
        self.common_space_learning_rate = common_space_learning_rate
        self.neg_triple_num = neg_triple_num
        self.cv_weight = cv_weight
        self.predicate_init_sim = predicate_init_sim
        self.predicate_soft_sim = predicate_soft_sim
        self.start_predicate_soft_alignment = start_predicate_soft_alignment
        self.min_predicate_triple_count = min_predicate_triple_count
        self.literal_encoder_model = literal_encoder_model
        self.literal_max_length = literal_max_length
        self.matching = matching
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
    ) -> MultiKELinker:
        """Train the three-view embedding and a shared space unifying them.

        Args:
            dataset1: Source entities -- also used to partition ``graph``
                into each KG's own triples.
            dataset2: Target entities. See ``dataset1``.
            ground_truth: List of ``(source_id, target_id)`` known-correct
                pairs -- MultiKE's only real supervision channel (see the
                module docstring's "cross-KG entity inference").
            graph: The combined relational structure of both KGs -- entity
                ids on both sides must already be disjoint.
            attribute_triples1: KG1's ``(entity_id, predicate, value)``
                attribute triples. **Required** -- see the module
                docstring's deviation note.
            attribute_triples2: KG2's attribute triples. **Required**.
            random_state: Seed for reproducible training (also seeds the
                literal encoder's random projection). Left unspecified,
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
            LinkingTKError: If ``attribute_triples1`` and
                ``attribute_triples2`` are both empty, or if none of
                ``ground_truth``'s pairs have both ids present in
                ``graph``'s own triples.
            OptionalDependencyError: If torch isn't installed.
        """
        if not attribute_triples1 and not attribute_triples2:
            raise LinkingTKError(
                "MultiKELinker.fit() needs attribute triples -- `attribute_triples1` "
                "and `attribute_triples2` are both empty, leaving nothing to train "
                "the attribute view or literal encoder with."
            )

        try:
            import torch
            import torch.nn.functional as functional
        except ImportError as exc:
            raise OptionalDependencyError("MultiKELinker", "kge") from exc

        if random_state is not None:
            torch.manual_seed(random_state)
        rng = np.random.default_rng(random_state)

        ids1 = {entity.id for entity in dataset1}
        ids2 = {entity.id for entity in dataset2}
        triples = to_triples(graph)
        triples1 = [t for t in triples if t[0] in ids1]
        triples2 = [t for t in triples if t[0] in ids2]

        entity_to_id, relation_to_id = build_id_mappings(triples)
        mapped1 = map_triples_to_ids(triples1, entity_to_id, relation_to_id)
        mapped2 = map_triples_to_ids(triples2, entity_to_id, relation_to_id)
        ctx1 = build_kg_context(mapped1)
        ctx2 = build_kg_context(mapped2)

        seed_pairs = [(s, t) for s, t in ground_truth if s in entity_to_id and t in entity_to_id]
        if not seed_pairs:
            raise LinkingTKError(
                "None of `ground_truth`'s pairs have both ids present in `graph`'s "
                "own triples; fit() has no seed pairs to train MultiKE's cross-KG "
                "entity inference with -- its only real supervision channel."
            )

        attr1_in = [t for t in attribute_triples1 if t[0] in entity_to_id]
        attr2_in = [t for t in attribute_triples2 if t[0] in entity_to_id]
        attr1_filtered, attr2_filtered = filter_frequent_predicates(
            attr1_in, attr2_in, self.min_predicate_triple_count
        )
        attr1_clean = self._clean_triples(attr1_filtered)
        attr2_clean = self._clean_triples(attr2_filtered)

        attribute_to_id = {
            predicate: index
            for index, predicate in enumerate(
                sorted({p for _, p, _ in attr1_clean} | {p for _, p, _ in attr2_clean})
            )
        }
        values = sorted({v for _, _, v in attr1_clean} | {v for _, _, v in attr2_clean})
        value_to_id = {value: index for index, value in enumerate(values)}

        def map_attribute_triples(mapped_triples: list[Triple]) -> npt.NDArray[np.int64]:
            rows = [
                (entity_to_id[e], attribute_to_id[a], value_to_id[v]) for e, a, v in mapped_triples
            ]
            return np.array(rows, dtype=np.int64) if rows else np.empty((0, 3), dtype=np.int64)

        attr_mapped1 = map_attribute_triples(attr1_clean)
        attr_mapped2 = map_attribute_triples(attr2_clean)

        aligned_relations = align_predicates_by_name(
            {p for _, p, _ in triples1}, {p for _, p, _ in triples2}, self.predicate_init_sim
        )
        aligned_attributes = align_predicates_by_name(
            {p for _, p, _ in attr1_clean}, {p for _, p, _ in attr2_clean}, self.predicate_init_sim
        )

        weighted1 = weight_attribute_triples(
            attr1_clean, aligned_attributes, self.predicate_soft_sim, is_kg1=True
        )
        weighted2 = weight_attribute_triples(
            attr2_clean, aligned_attributes, self.predicate_soft_sim, is_kg1=False
        )
        actx1 = build_attribute_context(
            attr_mapped1, np.array([w for *_, w in weighted1], dtype=np.float32)
        )
        actx2 = build_attribute_context(
            attr_mapped2, np.array([w for *_, w in weighted2], dtype=np.float32)
        )

        cross_rel1, cross_rel2 = generate_cross_kg_relation_triples(seed_pairs, triples1, triples2)
        cross_rel_ids = np.concatenate(
            [
                map_triples_to_ids(cross_rel1, entity_to_id, relation_to_id),
                map_triples_to_ids(cross_rel2, entity_to_id, relation_to_id),
            ],
            axis=0,
        )
        cross_attr1, cross_attr2 = generate_cross_kg_attribute_triples(
            seed_pairs, attr1_clean, attr2_clean
        )
        cross_attr_ids = np.concatenate(
            [map_attribute_triples(cross_attr1), map_attribute_triples(cross_attr2)], axis=0
        )

        pred_rel1_ids, pred_rel1_w = self._relation_ids_and_weights(
            substitute_predicate_triples(triples1, aligned_relations, is_kg1=True),
            entity_to_id,
            relation_to_id,
        )
        pred_rel2_ids, pred_rel2_w = self._relation_ids_and_weights(
            substitute_predicate_triples(triples2, aligned_relations, is_kg1=False),
            entity_to_id,
            relation_to_id,
        )
        pred_rel_ids = np.concatenate([pred_rel1_ids, pred_rel2_ids], axis=0)
        pred_rel_w = np.concatenate([pred_rel1_w, pred_rel2_w], axis=0)

        pred_attr1_ids, pred_attr1_w = self._attribute_ids_and_weights(
            substitute_predicate_triples(attr1_clean, aligned_attributes, is_kg1=True),
            entity_to_id,
            attribute_to_id,
            value_to_id,
        )
        pred_attr2_ids, pred_attr2_w = self._attribute_ids_and_weights(
            substitute_predicate_triples(attr2_clean, aligned_attributes, is_kg1=False),
            entity_to_id,
            attribute_to_id,
            value_to_id,
        )
        pred_attr_ids = np.concatenate([pred_attr1_ids, pred_attr2_ids], axis=0)
        pred_attr_w = np.concatenate([pred_attr1_w, pred_attr2_w], axis=0)

        name_embeds, literal_embeds = self._encode_literals(
            dataset1, dataset2, entity_to_id, values, random_state
        )
        name_embeds_t = torch.from_numpy(name_embeds)
        literal_embeds_t = torch.from_numpy(literal_embeds)

        rv_ent_embeds = self._init_embedding(torch, functional, len(entity_to_id))
        rel_embeds = self._init_embedding(torch, functional, max(1, len(relation_to_id)))
        av_ent_embeds = self._init_embedding(torch, functional, len(entity_to_id))
        attr_embeds = self._init_embedding_raw(torch, max(1, len(attribute_to_id)))
        ent_embeds = self._init_embedding(torch, functional, len(entity_to_id))
        scorer = build_attribute_scorer(self.embedding_dim)

        relation_optimizer = torch.optim.SGD(
            [rv_ent_embeds, rel_embeds, ent_embeds], lr=self.learning_rate
        )
        attribute_optimizer = torch.optim.SGD(
            [av_ent_embeds, attr_embeds, ent_embeds, *scorer.parameters()], lr=self.learning_rate
        )
        common_space_optimizer = torch.optim.SGD(
            [ent_embeds, rv_ent_embeds, av_ent_embeds], lr=self.common_space_learning_rate
        )
        cross_kg_relation_entity_optimizer = torch.optim.SGD(
            [rv_ent_embeds, rel_embeds], lr=self.learning_rate
        )
        cross_kg_attribute_entity_optimizer = torch.optim.SGD(
            [av_ent_embeds, attr_embeds, *scorer.parameters()], lr=self.learning_rate
        )
        cross_kg_relation_predicate_optimizer = torch.optim.SGD(
            [rv_ent_embeds, rel_embeds], lr=self.learning_rate
        )
        cross_kg_attribute_predicate_optimizer = torch.optim.SGD(
            [av_ent_embeds, attr_embeds, *scorer.parameters()], lr=self.learning_rate
        )

        entity_ids = np.arange(len(entity_to_id), dtype=np.int64)
        val_pairs = [
            (s, t) for s, t in (val_ground_truth or []) if s in entity_to_id and t in entity_to_id
        ]
        best_hits1 = -1.0
        epochs_without_improvement = 0

        for epoch in range(self.num_epochs):
            train_relation_view_epoch(
                rv_ent_embeds,
                rel_embeds,
                ent_embeds,
                name_embeds_t,
                relation_optimizer,
                ctx1,
                ctx2,
                rng,
                self.batch_size,
                self.neg_triple_num,
            )
            train_common_space_epoch(
                ent_embeds,
                rv_ent_embeds,
                av_ent_embeds,
                name_embeds_t,
                common_space_optimizer,
                entity_ids,
                rng,
                self.entity_batch_size,
                self.cv_weight,
            )
            train_cross_kg_relation_entity_epoch(
                rv_ent_embeds,
                rel_embeds,
                cross_kg_relation_entity_optimizer,
                cross_rel_ids,
                rng,
                self.batch_size,
            )
            if epoch >= self.start_predicate_soft_alignment:
                train_cross_kg_relation_predicate_epoch(
                    rv_ent_embeds,
                    rel_embeds,
                    cross_kg_relation_predicate_optimizer,
                    pred_rel_ids,
                    pred_rel_w,
                    rng,
                    self.batch_size,
                )

            train_attribute_view_epoch(
                av_ent_embeds,
                attr_embeds,
                literal_embeds_t,
                ent_embeds,
                name_embeds_t,
                scorer,
                attribute_optimizer,
                actx1,
                actx2,
                rng,
                self.attribute_batch_size,
            )
            train_common_space_epoch(
                ent_embeds,
                rv_ent_embeds,
                av_ent_embeds,
                name_embeds_t,
                common_space_optimizer,
                entity_ids,
                rng,
                self.entity_batch_size,
                self.cv_weight,
            )
            train_cross_kg_attribute_entity_epoch(
                av_ent_embeds,
                attr_embeds,
                literal_embeds_t,
                scorer,
                cross_kg_attribute_entity_optimizer,
                cross_attr_ids,
                rng,
                self.attribute_batch_size,
            )
            if epoch >= self.start_predicate_soft_alignment:
                train_cross_kg_attribute_predicate_epoch(
                    av_ent_embeds,
                    attr_embeds,
                    literal_embeds_t,
                    scorer,
                    cross_kg_attribute_predicate_optimizer,
                    pred_attr_ids,
                    pred_attr_w,
                    rng,
                    self.attribute_batch_size,
                )

            if val_pairs and (epoch + 1) % eval_every == 0:
                with torch.no_grad():
                    current_embeds = functional.normalize(ent_embeds, dim=1).numpy()
                hits1 = validation_hits1(current_embeds, entity_to_id, val_pairs)
                if hits1 <= best_hits1:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        break
                else:
                    best_hits1 = hits1
                    epochs_without_improvement = 0

        with torch.no_grad():
            final_embeds = functional.normalize(ent_embeds, dim=1).numpy()
        self._id_to_vector = {
            entity_id: final_embeds[index] for entity_id, index in entity_to_id.items()
        }
        self._fitted = True
        return self

    def _clean_triples(self, triples: list[Triple]) -> list[Triple]:
        cleaned = []
        for entity_id, predicate, value in triples:
            cleaned_value = clean_attribute_value(value)
            if cleaned_value:
                cleaned.append((entity_id, predicate, cleaned_value))
        return cleaned

    def _relation_ids_and_weights(
        self,
        quad: list[tuple[str, str, str, float]],
        entity_to_id: dict[str, int],
        relation_to_id: dict[str, int],
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
        if not quad:
            return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.float32)
        ids = np.array(
            [(entity_to_id[e], relation_to_id[p], entity_to_id[v]) for e, p, v, _ in quad],
            dtype=np.int64,
        )
        weights = np.array([w for *_, w in quad], dtype=np.float32)
        return ids, weights

    def _attribute_ids_and_weights(
        self,
        quad: list[tuple[str, str, str, float]],
        entity_to_id: dict[str, int],
        attribute_to_id: dict[str, int],
        value_to_id: dict[str, int],
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
        if not quad:
            return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.float32)
        ids = np.array(
            [(entity_to_id[e], attribute_to_id[p], value_to_id[v]) for e, p, v, _ in quad],
            dtype=np.int64,
        )
        weights = np.array([w for *_, w in quad], dtype=np.float32)
        return ids, weights

    def _encode_literals(
        self,
        dataset1: list[Entity],
        dataset2: list[Entity],
        entity_to_id: dict[str, int],
        values: list[str],
        random_state: int | None,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Build ``(name_embeds, literal_embeds)`` from one combined encoder call.

        Matches ``_generate_literal_vectors`` computing one shared
        ``literal_vectors_mat`` for entity names and attribute values
        together, then slicing it into the two tables.
        """
        entity_objects = {entity.id: entity for entity in dataset1 + dataset2}
        names_by_id: dict[str, str] = {}
        for entity_id in entity_to_id:
            entity = entity_objects.get(entity_id)
            if entity is not None and entity.labels:
                names_by_id[entity_id] = label_texts(entity)[0]
            else:
                names_by_id[entity_id] = label_from_raw(entity_id)

        literal_texts = sorted(set(names_by_id.values()) | set(values))
        literal_vectors = encode_literals(
            literal_texts,
            self.literal_encoder_model,
            self.embedding_dim,
            self.literal_max_length,
            random_state=random_state,
        )
        literal_index = {text: index for index, text in enumerate(literal_texts)}

        name_mat = np.zeros((len(entity_to_id), self.embedding_dim), dtype=np.float32)
        for entity_id, index in entity_to_id.items():
            name_mat[index] = literal_vectors[literal_index[names_by_id[entity_id]]]

        if values:
            literal_mat = np.stack([literal_vectors[literal_index[value]] for value in values])
        else:
            literal_mat = np.empty((0, self.embedding_dim), dtype=np.float32)

        return name_mat, literal_mat.astype(np.float32)

    def _init_embedding(self, torch: Any, functional: Any, size: int) -> Any:
        """Truncated-normal + L2-norm init, matching this family's `_init_embeddings` convention."""
        std = 1.0 / math.sqrt(self.embedding_dim)
        return torch.nn.Parameter(
            functional.normalize(
                torch.nn.init.trunc_normal_(
                    torch.empty(size, self.embedding_dim), std=std, a=-2 * std, b=2 * std
                ),
                dim=1,
            )
        )

    def _init_embedding_raw(self, torch: Any, size: int) -> Any:
        """Truncated-normal init, no L2-norm -- for ``attr_embeds`` only, see module docstring."""
        std = 1.0 / math.sqrt(self.embedding_dim)
        return torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(size, self.embedding_dim), std=std, a=-2 * std, b=2 * std
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
            raise LinkingTKError("MultiKELinker.link() called before fit().")

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
