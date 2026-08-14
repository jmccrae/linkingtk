"""KDCoE knowledge-graph-embedding linker for Entity Alignment.

Chen, M., Tian, Y., Chang, K.-W., Skiena, S., & Zaniolo, C. (2018).
Co-training Embeddings of Knowledge Graphs and Entity Descriptions for
Cross-lingual Entity Alignment. IJCAI 2018.
https://www.ijcai.org/proceedings/2018/0556.pdf

Like [MTransELinker][linkingtk.algorithms.ea.mtranse.MTransELinker] and
[IPTransELinker][linkingtk.algorithms.ea.iptranse.IPTransELinker], this is a
faithful reimplementation of KDCoE's actual training procedure, ported
directly from OpenEA's reference implementation
(https://github.com/nju-websoft/OpenEA/blob/master/src/openea/approaches/kdcoe.py
and its ``modules/base/losses.py``, ``modules/bootstrapping/alignment_finder.py``
support code) rather than from a from-scratch reading of the paper. KDCoE
**co-trains two signals with a bootstrapping loop between them**:

- A structural embedding: margin-based TransE (``margin=1.5``, one
  negative per positive, corrupting within each triple's own KG -- reuses
  ``_iptranse_training.py``'s ``sample_negative_triples``) bridged across
  KGs by a learned **mapping matrix**, the same "mapping"
  alignment module MTransE uses (*not* IPTransE's/JAPE's shared-id
  "sharing" trick) -- entities keep separate per-KG embedding rows in one
  combined table.
- A cross-lingual **entity-description encoder**: each entity's
  description text (or, when it has none, its own label -- see
  ``_kdcoe_text.extract_description_text``) is embedded via pretrained
  word vectors, then run through a GRU -> Conv1D -> attention-pool -> GRU
  -> attention-pool -> dense pipeline
  (``_kdcoe_torch.build_description_encoder``), trained with an in-batch
  contrastive loss that pulls each known-aligned pair's description
  vectors together and pushes apart other pairs in the same batch.

Each outer co-training iteration trains the description encoder to
convergence, uses it to find new high-confidence pairs among entities
outside the seed set (row-wise best-match above ``desc_sim_th``, reusing
``_iptranse_training.py``'s ``find_new_pairs`` -- confirmed to be exactly
OpenEA's own ``find_potential_alignment_greedily``/
``find_alignment`` logic), trains the structural+mapping model to
convergence (now also pulled toward those newly found pairs via a
*second*, ``new_param``-weighted mapping loss), and uses *it* to find new
pairs the same way -- stopping when a round re-discovers nothing beyond
what's already known (``max_co_training_iters`` caps the worst case).

**Key finding from reading the reference source, not obvious from the
paper**: `_get_desc_input` doesn't require every entity to have a real
description triple -- entities without one fall back to their own local
name as pseudo-description text. Since this repo's dataset loaders already
populate ``Entity.labels`` with exactly that fallback text (see
``linkingtk.datasets.openea_native``), the description pathway here always
has *some* text to work with. **Unlike
[JAPELinker][linkingtk.algorithms.ea.jape.JAPELinker], where omitting
``attribute_triples1``/``attribute_triples2`` disables the attribute
pathway entirely, here it only reduces the description signal to
label-only text** -- worth noting since it's a real interface difference
within this linker family.

**Word embeddings**: pretrained vectors are required (the description
encoder's whole point is leveraging generic-language priors that aren't
learnable from a 15K-entity alignment task alone). This fetches the same
file OpenEA's own published config uses --
``wiki-news-300d-1M.vec.zip`` from fastText's stable, Facebook-hosted URL
-- via ``linkingtk.datasets._util``'s ``fetch_cached``, the same mechanism
every dataset loader in this repo already uses. It's a much larger
one-time download (~681MB) than any other asset this repo fetches, so
``_kdcoe_text.py``'s ``load_fasttext_vectors`` streams the archived
``.vec`` member line-by-line and keeps only rows whose word is in this
run's own description vocabulary, rather than materializing the full
~1M-word file.

Deviations beyond what ``_kdcoe_text.py``/``_kdcoe_torch.py`` already
document:

- **Reference pool for bootstrapping**: OpenEA restricts the "not yet
  aligned" search pool to its own dataset loader's ``valid_entities +
  test_entities`` specifically. This repo's ``fit()`` only receives
  ``dataset1``/``dataset2`` and ``ground_truth`` (the train split), so the
  pool here is every entity *not* already a ``ground_truth`` source/target
  -- broader than OpenEA's pool, same already-established deviation as
  [IPTransELinker][linkingtk.algorithms.ea.iptranse.IPTransELinker]'s
  bootstrap pool and
  [JAPELinker][linkingtk.algorithms.ea.jape.JAPELinker]'s attribute
  reference pool (reuses ``_jape_training.py``'s ``reference_pools``
  directly).
- **Final scoring uses the structural+mapping embedding only**: the
  description encoder drives bootstrapping during ``fit()`` but doesn't
  contribute to ``link()``'s own similarity score -- ``link()``/
  ``_embedding()``/``_project()`` are near-identical to MTransE's.
- **Mapping-loss training samples with replacement** each step rather
  than OpenEA's ``random.sample`` (already-established MTransE deviation,
  same rationale).
- Early stopping uses this repo's own plain patience-counter Hits@1 check
  (twice -- once per phase, each reset every outer co-training iteration,
  matching OpenEA's own per-phase ``flag1``/``flag2`` reset) rather than
  OpenEA's exact two-flag ``early_stop()``.
- **Description-found and structurally-found bootstrap pairs are tracked
  and fed back in separately**, not merged into one shared set the way
  OpenEA's own ``self.new_alignment``/``self.new_alignment_index`` are.
  This is a deviation added after benchmarking on real EN-FR-15K-V1 data
  (#29): the description encoder's cosine similarities saturate near
  ``1.0`` for nearly every reference-pool pair regardless of correctness
  (median ~0.9998 in testing, so ``desc_sim_th=0.95`` accepts ~99.9% of
  the pool each round) -- ``find_new_pairs``'s row-argmax already
  discards *columns*, but here even each row's single best match isn't
  reliably correct at this data scale (row-argmax accuracy ~35%, and this
  dataset's real ``.../description`` triple coverage is sparse -- 10.5%
  of KG1 entities, 0.5% of KG2 -- so most descriptions are single-word
  label fallbacks, a weak signal). Feeding that many low-precision pairs
  into the structural mapping loss (even down-weighted by ``new_param``)
  collapsed structural Hits@1 from a clean ~0.40 to near-zero in testing.
  The description phase's own training tolerates being retrained on its
  own noisy self-found pairs without degrading (Hits@1 stayed ~0.44-0.46
  across iterations), so only structurally-*found* pairs (``sim_th``,
  whose similarity distribution isn't saturated the same way) feed the
  structural mapping-new loss; description-found pairs only feed the
  description phase's own next-iteration training.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from linkingtk.algorithms.base import DEFAULT_BLOCKING, BaseLinker
from linkingtk.algorithms.ea._device import resolve_device
from linkingtk.algorithms.ea._iptranse_training import find_new_pairs
from linkingtk.algorithms.ea._jape_training import reference_pools
from linkingtk.algorithms.ea._kdcoe_text import (
    build_word_ids,
    extract_description_text,
    load_fasttext_vectors,
    tokenize_description,
)
from linkingtk.algorithms.ea._kdcoe_torch import (
    KGContext,
    build_description_encoder,
    build_kg_context,
    train_desc_epoch,
    train_mapping_epoch,
    train_structural_epoch,
    validation_hits1_desc,
    validation_hits1_structural,
)
from linkingtk.algorithms.matching import DEFAULT_MATCHER, Matcher
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity
from linkingtk.core.result import AlignmentResult
from linkingtk.exceptions import LinkingTKError, OptionalDependencyError
from linkingtk.utils.graph import Graph, Triple, build_id_mappings, map_triples_to_ids, to_triples

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

_FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"


class KDCoELinker(BaseLinker):
    """Co-trains a structural TransE+mapping space with a description-text encoder.

    Must be [fit][linkingtk.algorithms.ea.kdcoe.KDCoELinker.fit] before
    [link][linkingtk.algorithms.base.BaseLinker.link] can be called. See
    the module docstring for how this differs from
    [MTransELinker][linkingtk.algorithms.ea.mtranse.MTransELinker] and
    [IPTransELinker][linkingtk.algorithms.ea.iptranse.IPTransELinker].

    Args:
        embedding_dim: Dimensionality of the trained entity/relation
            embeddings and the (square) mapping matrix. OpenEA's published
            EN-FR-15K-V1 config uses ``100``.
        num_epochs: Per-phase training-epoch cap (structural or
            description), each co-training iteration. OpenEA's config
            allows up to ``2000`` per phase with early stopping (see
            ``val_ground_truth`` on [fit][linkingtk.algorithms.ea.kdcoe.KDCoELinker.fit]).
        batch_size: Mini-batch size for structural triple-loss training;
            the mapping-loss batch size is derived from it (same number of
            minibatches per epoch for both, matching OpenEA).
        learning_rate: Adagrad's learning rate, for every optimizer here
            (structural, mapping, mapping-new, description) -- OpenEA's
            published value is ``0.01``.
        margin: Hinge margin for the structural triple loss. OpenEA's
            published value is ``1.5``.
        alpha: Weight applied to the seed-pairs mapping loss. OpenEA's
            published value is ``5.0`` (matches
            [MTransELinker][linkingtk.algorithms.ea.mtranse.MTransELinker]'s
            own default).
        new_param: Weight applied to the bootstrapped-pairs mapping loss.
            OpenEA's published value is ``0.1``.
        wv_dim: Pretrained word-vector dimensionality. Must match
            ``word_embed_url``'s file -- OpenEA's published fastText file
            is ``300``-dimensional.
        default_desc_length: Fixed word-sequence length each entity's
            description is padded/truncated to. OpenEA's published value
            is ``4``.
        desc_batch_size: Mini-batch size for description-encoder training.
            OpenEA's published value is ``512``.
        desc_sim_th: Minimum description-embedding similarity for a
            bootstrap round to accept a newly found pair. OpenEA's
            published value is ``0.95``.
        sim_th: Minimum (mapped) structural-embedding similarity for a
            bootstrap round to accept a newly found pair. OpenEA's
            published value is ``0.8``.
        max_co_training_iters: Cap on outer co-training iterations (each
            alternating a description-encoder phase and a structural
            phase). OpenEA's published value (``max_iter``) is ``5``.
        word_embed_url: URL of a fastText-format ``.vec.zip`` (see
            ``_kdcoe_text.py``'s ``load_fasttext_vectors``). Defaults to
            the real, published pretrained file; tests
            override this with a ``file://`` url pointing at a tiny local
            fixture.
        cache_dir: Passed through to ``fetch_cached`` for both the
            word-vector download and (indirectly, via callers'
            ``attribute_triples``) any dataset assets. ``None`` uses
            ``fetch_cached``'s own default cache location.
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
        alpha: float = 5.0,
        new_param: float = 0.1,
        wv_dim: int = 300,
        default_desc_length: int = 4,
        desc_batch_size: int = 512,
        desc_sim_th: float = 0.95,
        sim_th: float = 0.8,
        max_co_training_iters: int = 5,
        word_embed_url: str = _FASTTEXT_URL,
        cache_dir: Path | None = None,
        matching: Matcher = DEFAULT_MATCHER,
        device: str = "cpu",
    ) -> None:
        self.embedding_dim = embedding_dim
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.margin = margin
        self.alpha = alpha
        self.new_param = new_param
        self.wv_dim = wv_dim
        self.default_desc_length = default_desc_length
        self.desc_batch_size = desc_batch_size
        self.desc_sim_th = desc_sim_th
        self.sim_th = sim_th
        self.max_co_training_iters = max_co_training_iters
        self.word_embed_url = word_embed_url
        self.cache_dir = cache_dir
        self.matching = matching
        self.device = device
        self._id_to_vector: dict[str, npt.NDArray[np.floating[Any]]] = {}
        self._mapping: npt.NDArray[np.floating[Any]] | None = None
        self._fitted = False

    def fit(  # noqa: PLR0915 -- co-training loop genuinely needs this much sequential setup
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
    ) -> KDCoELinker:
        """Co-train the structural+mapping space and the description encoder.

        Args:
            dataset1: Source entities -- also used to partition ``graph``
                into each KG's own triples (same approach
                [IPTransELinker][linkingtk.algorithms.ea.iptranse.IPTransELinker]
                uses) and as the description pathway's label-fallback source.
            dataset2: Target entities. See ``dataset1``.
            ground_truth: List of ``(source_id, target_id)`` known-correct
                pairs, used as both phases' seed training pairs.
            graph: The combined relational structure of both KGs -- entity
                ids on both sides must already be disjoint.
            random_state: Seed for reproducible training. Left
                unspecified, training is non-deterministic.
            val_ground_truth: Optional held-out pairs used for early
                stopping in *both* phases -- every ``eval_every`` epochs,
                Hits@1 is checked, and that phase stops after ``patience``
                checks with no improvement. If ``None`` (default), each
                phase trains the full ``num_epochs`` unconditionally.
            patience: Number of non-improving ``eval_every``-spaced checks
                to tolerate before stopping a phase early. Only used if
                ``val_ground_truth`` is given.
            eval_every: How often (in epochs) to check ``val_ground_truth``.
                Only used if ``val_ground_truth`` is given.
            attribute_triples1: KG1's ``(entity_id, predicate, value)``
                attribute triples, if available -- entities with a
                ``.../description``-like predicate triple use its value as
                description text; others fall back to their own label
                (see the module docstring). May be omitted (or partial) --
                the description pathway still runs, just with less real
                description text.
            attribute_triples2: KG2's attribute triples. See
                ``attribute_triples1``.

        Returns:
            ``self``, for chaining.

        Raises:
            LinkingTKError: If none of ``ground_truth``'s pairs have both
                ids present in ``graph``'s own triples -- nothing to seed
                either phase's training with -- or if ``device`` is
                invalid or unavailable.
            OptionalDependencyError: If torch isn't installed.
        """
        try:
            import torch
            import torch.nn.functional as functional
        except ImportError as exc:
            raise OptionalDependencyError("KDCoELinker", "kge") from exc

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
        ctx1 = build_kg_context(mapped1)
        ctx2 = build_kg_context(mapped2)

        seed_pairs = [(s, t) for s, t in ground_truth if s in entity_to_id and t in entity_to_id]
        if not seed_pairs:
            raise LinkingTKError(
                "None of `ground_truth`'s pairs have both ids present in `graph`'s "
                "own triples; fit() has no seed pairs to train KDCoE's structural "
                "and description phases with."
            )
        seed_source_ids = np.array([entity_to_id[s] for s, _ in seed_pairs], dtype=np.int64)
        seed_target_ids = np.array([entity_to_id[t] for _, t in seed_pairs], dtype=np.int64)

        word_embeds, word_ids = self._build_description_inputs(
            dataset1, dataset2, attribute_triples1 or [], attribute_triples2 or [], device
        )

        entity_embeds, relation_embeds, mapping_mat, eye = self._init_structural(
            torch, functional, len(entity_to_id), len(relation_to_id), device
        )
        structural_optimizer = torch.optim.Adagrad(
            [entity_embeds, relation_embeds], lr=self.learning_rate
        )
        mapping_optimizer = torch.optim.Adagrad([mapping_mat], lr=self.learning_rate)
        mapping_optimizer_new = torch.optim.Adagrad([mapping_mat], lr=self.learning_rate)
        desc_encoder = build_description_encoder(self.wv_dim, self.default_desc_length).to(device)
        desc_optimizer = torch.optim.Adagrad(desc_encoder.parameters(), lr=self.learning_rate)

        val_pairs = [
            (s, t) for s, t in (val_ground_truth or []) if s in entity_to_id and t in entity_to_id
        ]
        pool1, pool2 = reference_pools(sorted(ids1), sorted(ids2), seed_pairs)

        desc_bootstrapped_pairs: set[tuple[str, str]] = set()
        rel_bootstrapped_pairs: set[tuple[str, str]] = set()
        triple_steps = max(1, math.ceil((len(ctx1.triples) + len(ctx2.triples)) / self.batch_size))

        for _iteration in range(self.max_co_training_iters):
            self._train_desc_phase(
                torch,
                desc_encoder,
                desc_optimizer,
                word_embeds,
                word_ids,
                seed_pairs + list(desc_bootstrapped_pairs),
                val_pairs,
                rng,
                patience,
                eval_every,
            )
            desc_found = self._find_new_pairs_via_desc(
                torch, desc_encoder, word_embeds, word_ids, pool1, pool2
            )
            if desc_found and desc_found <= desc_bootstrapped_pairs:
                break
            desc_bootstrapped_pairs |= desc_found

            bootstrap_source_ids = np.array(
                [entity_to_id[s] for s, _ in rel_bootstrapped_pairs], dtype=np.int64
            )
            bootstrap_target_ids = np.array(
                [entity_to_id[t] for _, t in rel_bootstrapped_pairs], dtype=np.int64
            )
            self._train_structural_phase(
                torch,
                functional,
                entity_embeds,
                relation_embeds,
                mapping_mat,
                eye,
                structural_optimizer,
                mapping_optimizer,
                mapping_optimizer_new,
                ctx1,
                ctx2,
                seed_source_ids,
                seed_target_ids,
                bootstrap_source_ids,
                bootstrap_target_ids,
                entity_to_id,
                val_pairs,
                rng,
                triple_steps,
                patience,
                eval_every,
            )
            rel_found = self._find_new_pairs_via_structural(
                torch, functional, entity_embeds, mapping_mat, entity_to_id, pool1, pool2
            )
            if rel_found and rel_found <= rel_bootstrapped_pairs:
                break
            rel_bootstrapped_pairs |= rel_found

        with torch.no_grad():
            final_embeds = functional.normalize(entity_embeds, dim=1).cpu().numpy()
            final_mapping = mapping_mat.cpu().numpy()
        self._id_to_vector = {
            entity_id: final_embeds[index] for entity_id, index in entity_to_id.items()
        }
        self._mapping = final_mapping
        self._fitted = True
        return self

    def _build_description_inputs(
        self,
        dataset1: list[Entity],
        dataset2: list[Entity],
        attribute_triples1: list[Triple],
        attribute_triples2: list[Triple],
        device: Any,
    ) -> tuple[Any, dict[str, npt.NDArray[np.int64]]]:
        """Fixed word-vector table plus every entity's fixed-length word-id sequence."""
        import torch

        entities = dataset1 + dataset2
        descriptions = extract_description_text(entities, attribute_triples1 + attribute_triples2)
        tokens_by_entity = {
            entity_id: tokenize_description(text) for entity_id, text in descriptions.items()
        }
        vocabulary = {token for tokens in tokens_by_entity.values() for token in tokens}
        word_vectors = load_fasttext_vectors(self.word_embed_url, vocabulary, self.cache_dir)

        vocabulary_words = sorted(word_vectors)
        word_to_index = {word: index for index, word in enumerate(vocabulary_words)}
        oov_id = len(word_to_index)
        rows = [word_vectors[word] for word in vocabulary_words]
        rows.append(np.zeros(self.wv_dim, dtype=np.float32))
        word_embeds_np = np.stack(rows).astype(np.float32)
        word_embeds = torch.from_numpy(word_embeds_np).float().to(device)

        word_ids = build_word_ids(tokens_by_entity, word_to_index, self.default_desc_length, oov_id)
        return word_embeds, word_ids

    def _init_structural(
        self, torch: Any, functional: Any, num_entities: int, num_relations: int, device: Any
    ) -> tuple[Any, Any, Any, Any]:
        """Unit-normalized init, matching OpenEA's ``init="unit"`` (same as MTransE's)."""
        entity_embeds = torch.nn.Parameter(
            functional.normalize(
                torch.randn(num_entities, self.embedding_dim, device=device), dim=1
            )
        )
        relation_embeds = torch.nn.Parameter(
            functional.normalize(
                torch.randn(num_relations, self.embedding_dim, device=device), dim=1
            )
        )
        mapping_mat = torch.nn.Parameter(
            torch.nn.init.orthogonal_(
                torch.empty(self.embedding_dim, self.embedding_dim, device=device)
            )
        )
        eye = torch.eye(self.embedding_dim, device=device)
        return entity_embeds, relation_embeds, mapping_mat, eye

    def _train_desc_phase(
        self,
        torch: Any,
        desc_encoder: Any,
        desc_optimizer: Any,
        word_embeds: Any,
        word_ids: dict[str, npt.NDArray[np.int64]],
        desc_pairs: list[tuple[str, str]],
        val_pairs: list[tuple[str, str]],
        rng: np.random.Generator,
        patience: int,
        eval_every: int,
    ) -> None:
        """Train the description encoder to convergence for one co-training iteration."""
        best_hits1 = -1.0
        epochs_without_improvement = 0
        for epoch in range(self.num_epochs):
            train_desc_epoch(
                desc_encoder,
                desc_optimizer,
                word_embeds,
                word_ids,
                desc_pairs,
                rng,
                self.desc_batch_size,
            )
            if val_pairs and (epoch + 1) % eval_every == 0:
                hits1 = validation_hits1_desc(desc_encoder, word_embeds, word_ids, val_pairs)
                if hits1 <= best_hits1:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        break
                else:
                    best_hits1 = hits1
                    epochs_without_improvement = 0

    def _train_structural_phase(
        self,
        torch: Any,
        functional: Any,
        entity_embeds: Any,
        relation_embeds: Any,
        mapping_mat: Any,
        eye: Any,
        structural_optimizer: Any,
        mapping_optimizer: Any,
        mapping_optimizer_new: Any,
        ctx1: KGContext,
        ctx2: KGContext,
        seed_source_ids: npt.NDArray[np.int64],
        seed_target_ids: npt.NDArray[np.int64],
        bootstrap_source_ids: npt.NDArray[np.int64],
        bootstrap_target_ids: npt.NDArray[np.int64],
        entity_to_id: dict[str, int],
        val_pairs: list[tuple[str, str]],
        rng: np.random.Generator,
        triple_steps: int,
        patience: int,
        eval_every: int,
    ) -> None:
        """Train the structural+mapping space to convergence for one co-training iteration."""
        best_hits1 = -1.0
        epochs_without_improvement = 0
        for epoch in range(self.num_epochs):
            train_structural_epoch(
                entity_embeds,
                relation_embeds,
                structural_optimizer,
                ctx1,
                ctx2,
                rng,
                self.batch_size,
                self.margin,
            )
            train_mapping_epoch(
                entity_embeds,
                mapping_mat,
                eye,
                mapping_optimizer,
                seed_source_ids,
                seed_target_ids,
                rng,
                triple_steps,
                self.alpha,
            )
            train_mapping_epoch(
                entity_embeds,
                mapping_mat,
                eye,
                mapping_optimizer_new,
                bootstrap_source_ids,
                bootstrap_target_ids,
                rng,
                triple_steps,
                self.new_param,
            )
            if val_pairs and (epoch + 1) % eval_every == 0:
                with torch.no_grad():
                    current_embeds = functional.normalize(entity_embeds, dim=1).cpu().numpy()
                    current_mapping = mapping_mat.cpu().numpy()
                hits1 = validation_hits1_structural(
                    current_embeds, current_mapping, entity_to_id, val_pairs
                )
                if hits1 <= best_hits1:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        break
                else:
                    best_hits1 = hits1
                    epochs_without_improvement = 0

    def _find_new_pairs_via_desc(
        self,
        torch: Any,
        desc_encoder: Any,
        word_embeds: Any,
        word_ids: dict[str, npt.NDArray[np.int64]],
        pool1: list[str],
        pool2: list[str],
    ) -> set[tuple[str, str]]:
        """Row-wise best-match-above-``desc_sim_th`` pairs among the reference pools."""
        if not pool1 or not pool2:
            return set()
        with torch.no_grad():
            device = word_embeds.device
            ids1 = np.stack([word_ids[e] for e in pool1])
            ids2 = np.stack([word_ids[e] for e in pool2])
            embeds1 = desc_encoder(word_embeds[torch.from_numpy(ids1).long().to(device)])
            embeds2 = desc_encoder(word_embeds[torch.from_numpy(ids2).long().to(device)])
            embeds1_np = embeds1.cpu().numpy()
            embeds2_np = embeds2.cpu().numpy()
        sim_mat = embeds1_np @ embeds2_np.T
        found = find_new_pairs(sim_mat, self.desc_sim_th)
        return {(pool1[i], pool2[j]) for i, j, _ in found}

    def _find_new_pairs_via_structural(
        self,
        torch: Any,
        functional: Any,
        entity_embeds: Any,
        mapping_mat: Any,
        entity_to_id: dict[str, int],
        pool1: list[str],
        pool2: list[str],
    ) -> set[tuple[str, str]]:
        """Row-wise best-match-above-``sim_th`` pairs among the reference pools."""
        if not pool1 or not pool2:
            return set()
        with torch.no_grad():
            current = functional.normalize(entity_embeds, dim=1).cpu().numpy()
            current_mapping = mapping_mat.cpu().numpy()
        pool1_ids = np.array([entity_to_id[e] for e in pool1], dtype=np.int64)
        pool2_ids = np.array([entity_to_id[e] for e in pool2], dtype=np.int64)
        mapped1 = current[pool1_ids] @ current_mapping
        sim_mat = mapped1 @ current[pool2_ids].T
        found = find_new_pairs(sim_mat, self.sim_th)
        return {(pool1[i], pool2[j]) for i, j, _ in found}

    def link(
        self,
        dataset1: list[Entity],
        dataset2: list[Entity],
        graph: Graph = None,
        blocking: BlockingStrategy = DEFAULT_BLOCKING,
    ) -> list[AlignmentResult]:
        if not self._fitted:
            raise LinkingTKError("KDCoELinker.link() called before fit().")

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
        """Vector used to score ``entity_id`` as a scored pair's source side (projected)."""
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
