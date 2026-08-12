"""MTransE knowledge-graph-embedding linker for Entity Alignment.

Chen, M., Tian, Y., Yang, M., & Zaniolo, C. (2017). Multilingual Knowledge
Graph Embeddings for Cross-lingual Knowledge Alignment. IJCAI 2017.
https://www.ijcai.org/proceedings/2017/0209.pdf

Unlike [KGELinker][linkingtk.algorithms.ea.kge.KGELinker]'s simplified
single-shared-pseudo-relation trick, this is a faithful reimplementation of
MTransE's actual training procedure, ported directly from OpenEA's
reference implementation
(https://github.com/nju-websoft/OpenEA/blob/master/src/openea/approaches/mtranse.py
and .../modules/base/mapping.py) rather than from a from-scratch reading of
the paper, since OpenEA's actual algorithm differs from a naive paper
reading in ways that matter for reproducing its published numbers:

- Entities and relations from *both* KGs share one embedding table (not
  two independently-trained per-KG spaces). Since each KG's own triples
  only ever reference that KG's own entities, this has the same practical
  effect as two separate spaces, but reuses this repo's existing
  combined-graph id-mapping utilities directly.
- The embedding step uses pure positive-triple loss (``sum(||h+r-t||^2)``
  over real triples only) with **no negative sampling** -- unlike
  ``KGELinker``/pykeen's ``SLCWATrainingLoop``. This is why training is
  hand-rolled in PyTorch here rather than routed through pykeen.
- Entities and relations are L2-normalized to the unit sphere on *every*
  forward pass (not just at initialization) -- this is what keeps the
  no-negative-sampling loss from collapsing to the trivial zero solution.
- Alignment signal comes entirely from a learned square mapping matrix
  (orthogonally initialized, trained via its own optimizer against a
  soft-orthogonality-regularized loss over seed pairs), not from injecting
  pseudo-triples into the embedding space itself.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from linkingtk.algorithms.base import DEFAULT_BLOCKING, BaseLinker
from linkingtk.algorithms.matching import DEFAULT_MATCHER, Matcher
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity
from linkingtk.core.result import AlignmentResult
from linkingtk.exceptions import LinkingTKError, OptionalDependencyError
from linkingtk.utils.graph import Graph, build_id_mappings, map_triples_to_ids, to_triples

if TYPE_CHECKING:
    import numpy.typing as npt


class MTransELinker(BaseLinker):
    """Scores candidate pairs via a learned cross-KG mapping over shared TransE embeddings.

    Must be [fit][linkingtk.algorithms.ea.mtranse.MTransELinker.fit] before
    [link][linkingtk.algorithms.base.BaseLinker.link] can be called.

    Args:
        embedding_dim: Dimensionality of the trained entity/relation
            embeddings and the (square) mapping matrix. OpenEA's published
            EN-FR-15K-V1 config uses ``100``.
        num_epochs: Training epochs, each alternating one epoch of
            triple-loss training with one epoch of mapping-loss training.
            OpenEA's config allows up to ``2000`` with early stopping (see
            ``val_ground_truth`` on [fit][linkingtk.algorithms.ea.mtranse.MTransELinker.fit]).
        batch_size: Mini-batch size for triple-loss training; the
            per-epoch mapping-loss batch size is derived from it (matching
            OpenEA: the same number of mini-batches per epoch for both).
        learning_rate: Adagrad's learning rate, for both the embedding and
            mapping optimizers -- OpenEA's published value is ``0.01``.
        alpha: Weight applied to the mapping loss (approximation error
            plus orthogonality regularization). OpenEA's published value
            is ``5.0``.
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
        alpha: float = 5.0,
        matching: Matcher = DEFAULT_MATCHER,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.matching = matching
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
    ) -> MTransELinker:
        """Train shared TransE embeddings plus a learned cross-KG mapping.

        Args:
            dataset1: Source entities. Unused beyond mirroring
                [KGELinker.fit][linkingtk.algorithms.ea.kge.KGELinker.fit]'s
                signature -- which entities get embeddings is determined
                entirely by ``graph``.
            dataset2: Target entities. See ``dataset1``.
            ground_truth: List of ``(source_id, target_id)`` known-correct
                pairs used to train the mapping matrix. Unlike
                ``KGELinker``, these are *not* injected into ``graph`` as
                extra triples.
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
                the mapping with.
            OptionalDependencyError: If torch isn't installed.
        """
        try:
            import torch
            import torch.nn.functional as functional
        except ImportError as exc:
            raise OptionalDependencyError("MTransELinker", "kge") from exc

        if random_state is not None:
            torch.manual_seed(random_state)

        triples = to_triples(graph)
        entity_to_id, relation_to_id = build_id_mappings(triples)
        mapped_triples = map_triples_to_ids(triples, entity_to_id, relation_to_id)
        heads = torch.from_numpy(mapped_triples[:, 0]).long()
        rels = torch.from_numpy(mapped_triples[:, 1]).long()
        tails = torch.from_numpy(mapped_triples[:, 2]).long()

        seed_pairs = [(s, t) for s, t in ground_truth if s in entity_to_id and t in entity_to_id]
        if not seed_pairs:
            raise LinkingTKError(
                "None of `ground_truth`'s pairs have both ids present in `graph`'s "
                "own triples; fit() has no seed pairs to train MTransE's mapping with."
            )
        seed_source = torch.tensor([entity_to_id[s] for s, _ in seed_pairs])
        seed_target = torch.tensor([entity_to_id[t] for _, t in seed_pairs])

        entity_embeds = torch.nn.Parameter(
            functional.normalize(torch.randn(len(entity_to_id), self.embedding_dim), dim=1)
        )
        relation_embeds = torch.nn.Parameter(
            functional.normalize(torch.randn(len(relation_to_id), self.embedding_dim), dim=1)
        )
        mapping_mat = torch.nn.Parameter(
            torch.nn.init.orthogonal_(torch.empty(self.embedding_dim, self.embedding_dim))
        )
        eye = torch.eye(self.embedding_dim)

        triple_optimizer = torch.optim.Adagrad(
            [entity_embeds, relation_embeds], lr=self.learning_rate
        )
        mapping_optimizer = torch.optim.Adagrad([mapping_mat], lr=self.learning_rate)

        triple_steps = max(1, math.ceil(len(heads) / self.batch_size))
        seed_batch_size = max(1, len(seed_source) // triple_steps)

        val_pairs = [
            (s, t) for s, t in (val_ground_truth or []) if s in entity_to_id and t in entity_to_id
        ]
        best_hits1 = -1.0
        epochs_without_improvement = 0

        for epoch in range(self.num_epochs):
            perm = torch.randperm(len(heads))
            for step in range(triple_steps):
                idx = perm[step * self.batch_size : (step + 1) * self.batch_size]
                h = functional.normalize(entity_embeds[heads[idx]], dim=1)
                r = functional.normalize(relation_embeds[rels[idx]], dim=1)
                t = functional.normalize(entity_embeds[tails[idx]], dim=1)
                triple_loss = ((h + r - t) ** 2).sum()
                triple_optimizer.zero_grad()
                triple_loss.backward()  # type: ignore[no-untyped-call]
                triple_optimizer.step()

            for _step in range(triple_steps):
                batch_idx = torch.randint(0, len(seed_source), (seed_batch_size,))
                s_vecs = functional.normalize(entity_embeds[seed_source[batch_idx]], dim=1)
                t_vecs = functional.normalize(entity_embeds[seed_target[batch_idx]], dim=1)
                map_loss = ((t_vecs - s_vecs @ mapping_mat) ** 2).sum()
                orth_loss = ((mapping_mat @ mapping_mat.T - eye) ** 2).sum()
                mapping_loss = self.alpha * (map_loss + orth_loss)
                mapping_optimizer.zero_grad()
                mapping_loss.backward()  # type: ignore[no-untyped-call]
                mapping_optimizer.step()

            if val_pairs and (epoch + 1) % eval_every == 0:
                with torch.no_grad():
                    current_embeds = functional.normalize(entity_embeds, dim=1).numpy()
                    current_mapping = mapping_mat.numpy()
                hits1 = _validation_hits1(current_embeds, current_mapping, entity_to_id, val_pairs)
                if hits1 <= best_hits1:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        break
                else:
                    best_hits1 = hits1
                    epochs_without_improvement = 0

        with torch.no_grad():
            final_embeds = functional.normalize(entity_embeds, dim=1).numpy()
            final_mapping = mapping_mat.numpy()
        self._id_to_vector = {
            entity_id: final_embeds[index] for entity_id, index in entity_to_id.items()
        }
        self._mapping = final_mapping
        self._fitted = True
        return self

    def link(
        self,
        dataset1: list[Entity],
        dataset2: list[Entity],
        graph: Graph = None,
        blocking: BlockingStrategy = DEFAULT_BLOCKING,
    ) -> list[AlignmentResult]:
        if not self._fitted:
            raise LinkingTKError("MTransELinker.link() called before fit().")

        pairs = blocking.candidate_pairs(dataset1, dataset2)
        target_ids_by_source: dict[str, list[str]] = defaultdict(list)
        for entity1, entity2 in pairs:
            target_ids_by_source[entity1.id].append(entity2.id)

        candidates_by_source: dict[str, list[tuple[str, float]]] = {}
        for source_id, target_ids in target_ids_by_source.items():
            source_vector = self._project(self._embedding(source_id)).reshape(1, -1)
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

    def _project(self, vector: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
        assert self._mapping is not None  # noqa: S101 -- guarded by self._fitted in link()
        return vector @ self._mapping


def _validation_hits1(
    embeds: npt.NDArray[np.floating[Any]],
    mapping: npt.NDArray[np.floating[Any]],
    entity_to_id: dict[str, int],
    val_pairs: list[tuple[str, str]],
) -> float:
    """Cheap top-1 match rate over ``val_pairs``, for early-stopping only.

    A deliberately simplified stand-in for OpenEA's own dual-flag
    ``early_stop()`` (``modules/finding/evaluation.py``) -- a plain
    patience counter over Hits@1 rather than their exact two-flag
    comparison, computed directly via numpy rather than the full
    ``Matcher``/``BlockingStrategy`` pipeline, since this only needs a
    cheap scalar training signal, not ``AlignmentResult``s.
    """
    sources = [s for s, _ in val_pairs]
    targets = [t for _, t in val_pairs]
    source_matrix = np.stack([embeds[entity_to_id[s]] for s in sources]) @ mapping
    target_matrix = np.stack([embeds[entity_to_id[t]] for t in targets])
    similarities = cosine_similarity(source_matrix, target_matrix)
    predicted = np.argmax(similarities, axis=1)
    correct = sum(1 for i, j in enumerate(predicted) if j == i)
    return correct / len(val_pairs)
