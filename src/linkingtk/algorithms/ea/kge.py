"""Knowledge-graph-embedding linker for Entity Alignment.

Wraps pykeen (https://github.com/pykeen/pykeen) translational and bilinear
embedding models -- TransE, DistMult, RotatE by default, per DESIGN.md's
pykeen/OpenEA/EntMatcher references -- to score candidate pairs by
embedding similarity rather than string/feature similarity. pykeen has no
built-in Entity Alignment mode (it's a link-prediction library); the
approach here is a simplified version of the MTransE/IPTransE family's
core idea, not a reimplementation of either paper's full method (no
separate per-KG embedding spaces plus a learned transform, no triple
-swapping): both KGs' triples are combined into a single graph, the known
training alignment pairs are added to it as extra ``(source_id,
"__seed_alignment__", target_id)`` triples, and a single KGE model is
trained jointly over all of it -- pulling aligned (and structurally
analogous) entities' embeddings together via that one pseudo-relation.
"""

from __future__ import annotations

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
from linkingtk.utils.graph import Graph, to_triples

if TYPE_CHECKING:
    import numpy.typing as npt

_SEED_RELATION = "__seed_alignment__"


class KGELinker(BaseLinker):
    """Scores candidate pairs by trained knowledge-graph-embedding similarity.

    Must be :meth:`fit` before :meth:`link` can be called. Unlike
    :class:`~linkingtk.algorithms.feature_classifier.FeatureClassifierLinker`,
    training needs the KGs' relational structure, not just entity text --
    see :meth:`fit`'s ``graph`` argument.

    Args:
        model: A model name from pykeen's registry
            (``pykeen.models.model_resolver``, ~40 entries), e.g.
            ``"TransE"``, ``"DistMult"``, ``"RotatE"``. Not restricted to
            those three -- pykeen already provides its own named-model
            resolution, so there's nothing to add on top of it here -- but
            they're what this class is validated against; other models
            are best-effort, since not every pykeen model shares the same
            constructor keyword shape (e.g. some need more than
            ``embedding_dim``).
        embedding_dim: Dimensionality of the trained entity/relation
            embeddings.
        num_epochs: Training epochs.
        batch_size: Training batch size. Defaults to pykeen's own
            auto-selection (``None``).
        learning_rate: Adam's learning rate. pykeen's own default
            (``0.001``, via ``torch.optim.Adam``'s default) converges too
            slowly for the seed-alignment triples to pull related entities
            together within a practical epoch budget at small scale;
            ``0.01`` converges reliably in testing.
        matching: Strategy used to resolve scored candidates into final
            links. Defaults to
            `~linkingtk.algorithms.matching.GreedyMatcher`.
            `~linkingtk.algorithms.matching.OptimalMatcher`'s dense
            Hungarian-algorithm cost matrix is explicitly not meant for
            KG-scale entity counts (see its docstring) -- avoid it here
            for anything but small graphs.
    """

    def __init__(
        self,
        model: str = "TransE",
        embedding_dim: int = 50,
        num_epochs: int = 50,
        batch_size: int | None = None,
        learning_rate: float = 0.01,
        matching: Matcher = DEFAULT_MATCHER,
    ) -> None:
        self.model = model
        self.embedding_dim = embedding_dim
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
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
    ) -> KGELinker:
        """Train KGE embeddings jointly over ``graph`` plus seed alignment triples.

        Args:
            dataset1: Source entities.
            dataset2: Target entities.
            ground_truth: List of ``(source_id, target_id)`` known-correct
                pairs, added to the training graph as seed alignment
                triples (see the module docstring).
            graph: The combined relational structure of both KGs (e.g.
                ``to_triples(graph1) + to_triples(graph2)`` from a
                :class:`~linkingtk.datasets.base.GraphDatasetLoader`'s
                ``load_graphs()``) -- entity ids on both sides must
                already be disjoint, as they are from that loader.
            random_state: Seed for reproducible training. Left
                unspecified, pykeen trains non-deterministically.

        Returns:
            ``self``, for chaining.

        Raises:
            LinkingTKError: If no ``ground_truth`` pair's ids are actually
                present in ``dataset1``/``dataset2`` -- nothing to seed
                training with.
            OptionalDependencyError: If pykeen isn't installed.
        """
        ids1 = {entity.id for entity in dataset1}
        ids2 = {entity.id for entity in dataset2}
        seed_pairs = [(s, t) for s, t in ground_truth if s in ids1 and t in ids2]
        if not seed_pairs:
            raise LinkingTKError(
                "None of `ground_truth`'s pairs have ids present in `dataset1`/"
                "`dataset2`; fit() has no seed alignment triples to train with."
            )

        try:
            import torch
            from pykeen.models import model_resolver
            from pykeen.training import SLCWATrainingLoop
            from pykeen.triples import TriplesFactory
            from pykeen.utils import set_random_seed
            from torch.optim import Adam
        except ImportError as exc:
            raise OptionalDependencyError("KGELinker", "kge") from exc

        if random_state is not None:
            set_random_seed(random_state)

        triples = to_triples(graph) + [(s, _SEED_RELATION, t) for s, t in seed_pairs]
        triples_factory = TriplesFactory.from_labeled_triples(np.array(triples, dtype=str))
        model: Any = model_resolver.make(
            self.model,
            triples_factory=triples_factory,
            embedding_dim=self.embedding_dim,
            random_seed=random_state,
        )
        training_loop = SLCWATrainingLoop(
            model=model,
            triples_factory=triples_factory,
            optimizer=Adam(params=model.get_grad_params(), lr=self.learning_rate),
        )
        training_loop.train(
            triples_factory=triples_factory,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            use_tqdm=False,
        )

        model.eval()
        with torch.no_grad():
            embeddings = model.entity_representations[0](indices=None)
        if embeddings.is_complex():
            embeddings = torch.view_as_real(embeddings).reshape(embeddings.shape[0], -1)
        embeddings_array = embeddings.detach().numpy()
        self._id_to_vector = {
            entity_id: embeddings_array[index]
            for entity_id, index in triples_factory.entity_to_id.items()
        }
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
            raise LinkingTKError("KGELinker.link() called before fit().")

        pairs = blocking.candidate_pairs(dataset1, dataset2)
        target_ids_by_source: dict[str, list[str]] = defaultdict(list)
        for entity1, entity2 in pairs:
            target_ids_by_source[entity1.id].append(entity2.id)

        # One batched cosine_similarity call per source entity rather than
        # per pair -- sklearn's per-call validation/normalization overhead
        # otherwise dominates at real (post-blocking, still
        # many-candidates-per-source) KG scale.
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
