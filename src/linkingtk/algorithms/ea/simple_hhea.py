"""Simple-HHEA linker for Entity Alignment.

Ports Simple-HHEA (Jiang et al. 2024, "Toward Practical Entity Alignment
Method Design: Insights from New Highly Heterogeneous Knowledge Graph
Datasets", https://dl.acm.org/doi/pdf/10.1145/3589334.3645720) from its
reference implementation at https://github.com/DataArcTech/Simple-HHEA --
a base embedding method for highly heterogeneous KG pairs (e.g. an event
KG aligned to Wikipedia), fusing ALBERT name embeddings (whitened),
node2vec structural embeddings, and (optionally) Time2Vec temporal
embeddings through a small MLP trained with margin ranking loss.

Filed as #62, a prerequisite for #22 (ChatEA): the ChatEA paper's
published llama2-13b numbers are ChatEA layered on top of Simple-HHEA
specifically, not a generic KGE method, so `SimpleHHEALinker` is what
`ChatEALinker` will use as its `base_linker` for a genuine comparison.

See `linkingtk.algorithms.ea._simple_hhea_name`/`_simple_hhea_structure`/
`_simple_hhea_torch`/`_simple_hhea_training` for the individual feature-
group ports and their documented fidelity decisions vs. the reference.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from linkingtk.algorithms.base import DEFAULT_BLOCKING, BaseLinker
from linkingtk.algorithms.ea._simple_hhea_name import compute_kernel_bias, embed_names, whiten
from linkingtk.algorithms.ea._simple_hhea_structure import build_structure_embeddings
from linkingtk.algorithms.ea._simple_hhea_torch import SimpleHHEAModel, build_time_histogram
from linkingtk.algorithms.ea._simple_hhea_training import train as train_simple_hhea
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity
from linkingtk.core.result import AlignmentResult
from linkingtk.core.source import EntitySource
from linkingtk.exceptions import LinkingTKError
from linkingtk.matchers import DEFAULT_MATCHER, Matcher
from linkingtk.utils.device import resolve_device
from linkingtk.utils.graph import Graph, to_triples

if TYPE_CHECKING:
    import numpy.typing as npt

TemporalTriple = tuple[str, str, str, str | None, str | None]


class SimpleHHEALinker(BaseLinker):
    """Fuses whitened name embeddings, node2vec structural embeddings, and
    (optionally) Time2Vec temporal embeddings for Entity Alignment.

    Must be [fit][linkingtk.algorithms.ea.simple_hhea.SimpleHHEALinker.fit]
    before `link`. Exposes `source_embedding`/`target_embedding` like
    [KGELinker][linkingtk.algorithms.ea.kge.KGELinker], so it's a drop-in
    for [rank_exhaustive][linkingtk.eval.ranking.rank_exhaustive] -- and,
    once #22 (ChatEA) lands, a candidate `base_linker` for it too.

    Args:
        name_model: Hugging Face model id for name embedding, loaded via
            `AutoModel`/`AutoTokenizer.from_pretrained`. Defaults to the
            reference's own `albert-base-v2`.
        emb_size: Final fused embedding dimension.
        structure_size: Post-projection size of the structure branch
            before concatenation into the final embedding.
        time_size: Post-projection size of the time branch before
            concatenation into the final embedding.
        use_structure: Whether to compute and use node2vec structural
            embeddings at all (the reference's own `--no_structure`).
        walk_length: node2vec walk hyperparameter -- see
            [build_structure_embeddings][linkingtk.algorithms.ea._simple_hhea_structure.build_structure_embeddings].
        num_walks: node2vec walk hyperparameter -- see
            [build_structure_embeddings][linkingtk.algorithms.ea._simple_hhea_structure.build_structure_embeddings].
        structure_dim: node2vec walk hyperparameter -- see
            [build_structure_embeddings][linkingtk.algorithms.ea._simple_hhea_structure.build_structure_embeddings].
        max_degree: node2vec walk hyperparameter -- see
            [build_structure_embeddings][linkingtk.algorithms.ea._simple_hhea_structure.build_structure_embeddings].
        structure_workers: Worker threads for the structural Word2Vec
            fit -- see
            [build_structure_embeddings][linkingtk.algorithms.ea._simple_hhea_structure.build_structure_embeddings].
            Walk simulation itself is always single-threaded, matching
            the reference's own default invocation.
        structure_epochs: Word2Vec training epochs for the structural
            skip-gram model -- see
            [build_structure_embeddings][linkingtk.algorithms.ea._simple_hhea_structure.build_structure_embeddings].
        num_epochs: Training hyperparameter -- see
            [train][linkingtk.algorithms.ea._simple_hhea_training.train].
        learning_rate: Training hyperparameter -- see
            [train][linkingtk.algorithms.ea._simple_hhea_training.train].
        weight_decay: Training hyperparameter -- see
            [train][linkingtk.algorithms.ea._simple_hhea_training.train].
        gamma: Training hyperparameter -- see
            [train][linkingtk.algorithms.ea._simple_hhea_training.train].
        matching: Strategy used to resolve scored candidates into final
            links. Defaults to
            [GreedyMatcher][linkingtk.matchers.greedy.GreedyMatcher].
        device: Torch device to train/embed on, e.g. `"cpu"` (default) or
            `"cuda"`.
    """

    def __init__(
        self,
        name_model: str = "albert-base-v2",
        emb_size: int = 64,
        structure_size: int = 8,
        time_size: int = 8,
        use_structure: bool = True,
        walk_length: int = 80,
        num_walks: int = 10,
        structure_dim: int = 64,
        max_degree: int = 1000,
        structure_workers: int = 1,
        structure_epochs: int = 5,
        num_epochs: int = 1500,
        learning_rate: float = 0.01,
        weight_decay: float = 0.001,
        gamma: float = 1.0,
        matching: Matcher = DEFAULT_MATCHER,
        device: str = "cpu",
    ) -> None:
        self.name_model = name_model
        self.emb_size = emb_size
        self.structure_size = structure_size
        self.time_size = time_size
        self.use_structure = use_structure
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.structure_dim = structure_dim
        self.max_degree = max_degree
        self.structure_workers = structure_workers
        self.structure_epochs = structure_epochs
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gamma = gamma
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
        temporal_triples: list[TemporalTriple] | None = None,
        random_state: int | None = None,
    ) -> SimpleHHEALinker:
        """Trains the fusion model over `dataset1`/`dataset2`'s entities.

        Args:
            dataset1: Source entities.
            dataset2: Target entities.
            ground_truth: Known-correct `(source_id, target_id)` pairs.
                Doubles as both the margin-ranking training signal (like
                [KGELinker.fit][linkingtk.algorithms.ea.kge.KGELinker.fit]'s
                `ground_truth`) and the node2vec node-merge seed (see
                [build_structure_embeddings][linkingtk.algorithms.ea._simple_hhea_structure.build_structure_embeddings]'s
                `train_pairs`).
            graph: Combined relational structure of both KGs (e.g.
                `to_triples(graph1) + to_triples(graph2)`).
            temporal_triples: Optional `(subject_id, relation_id,
                object_id, start_label, end_label)` temporal facts (e.g.
                from
                [IcewsWikiDataset.load_temporal_graphs][linkingtk.datasets.icews.IcewsWikiDataset.load_temporal_graphs]),
                `start_label`/`end_label` as `"YYYY-MM"` strings or
                `None`. `None` (default) disables the time-feature branch
                entirely -- correct for non-ICEWS datasets, which have no
                temporal facts at all.
            random_state: Seed for reproducible training.

        Returns:
            `self`, for chaining.

        Raises:
            LinkingTKError: If no `ground_truth` pair's ids are actually
                present in `dataset1`/`dataset2`.
        """
        ids1 = {entity.id for entity in dataset1}
        ids2 = {entity.id for entity in dataset2}
        train_pairs = [(s, t) for s, t in ground_truth if s in ids1 and t in ids2]
        if not train_pairs:
            raise LinkingTKError(
                "None of `ground_truth`'s pairs have ids present in `dataset1`/"
                "`dataset2`; fit() has no training pairs."
            )

        if random_state is not None:
            torch.manual_seed(random_state)

        entities = dataset1 + dataset2
        entity_ids = [entity.id for entity in entities]
        entity_to_index = {entity_id: index for index, entity_id in enumerate(entity_ids)}

        raw_name_emb = embed_names(entities, model_name=self.name_model, device=self.device)
        kernel, bias = compute_kernel_bias(raw_name_emb, self.emb_size)
        name_emb = whiten(raw_name_emb, kernel, bias)

        structure_emb = None
        if self.use_structure:
            triples = to_triples(graph)
            structure_by_id = build_structure_embeddings(
                entity_ids,
                triples,
                train_pairs,
                dimensions=self.structure_dim,
                walk_length=self.walk_length,
                num_walks=self.num_walks,
                max_degree=self.max_degree,
                workers=self.structure_workers,
                epochs=self.structure_epochs,
                random_state=random_state,
            )
            structure_emb = np.stack([structure_by_id[entity_id] for entity_id in entity_ids])

        time_emb = (
            build_time_histogram(entity_ids, temporal_triples)
            if temporal_triples is not None
            else None
        )

        resolved_device = resolve_device(self.device)
        model = SimpleHHEAModel(
            name_emb,
            time_emb,
            structure_emb,
            emb_size=self.emb_size,
            structure_size=self.structure_size,
            time_size=self.time_size,
            device=resolved_device,
        ).to(resolved_device)

        train_pairs_idx = np.array(
            [(entity_to_index[s], entity_to_index[t]) for s, t in train_pairs], dtype=np.int64
        )
        train_simple_hhea(
            model,
            train_pairs_idx,
            len(entity_ids),
            num_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            gamma=self.gamma,
            random_state=random_state,
        )

        model.eval()
        with torch.no_grad():
            embeddings = model().cpu().numpy()
        self._id_to_vector = {
            entity_id: embeddings[index] for index, entity_id in enumerate(entity_ids)
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
            raise LinkingTKError("SimpleHHEALinker.link() called before fit().")

        pairs = blocking.candidate_pairs(dataset1, dataset2)
        target_ids_by_source: dict[str, list[str]] = defaultdict(list)
        for entity1, entity2 in pairs:
            target_ids_by_source[entity1.id].append(entity2.id)

        candidates_by_source: dict[str, list[tuple[str, float]]] = {}
        for source_id, target_ids in target_ids_by_source.items():
            source_vector = self.source_embedding(source_id).reshape(1, -1)
            target_matrix = np.stack([self.target_embedding(t) for t in target_ids])
            scores = cosine_similarity(source_vector, target_matrix)[0]
            candidates_by_source[source_id] = list(
                zip(target_ids, (float(score) for score in scores), strict=True)
            )

        return self.matching.match(candidates_by_source)

    def source_embedding(self, entity_id: str) -> npt.NDArray[np.floating[Any]]:
        """Vector used to score `entity_id` as a scored pair's source side."""
        return self._embedding(entity_id)

    def target_embedding(self, entity_id: str) -> npt.NDArray[np.floating[Any]]:
        """Vector used to score `entity_id` as a scored pair's target side."""
        return self._embedding(entity_id)

    def _embedding(self, entity_id: str) -> npt.NDArray[np.floating[Any]]:
        vector = self._id_to_vector.get(entity_id)
        if vector is None:
            raise LinkingTKError(
                f"Entity {entity_id!r} has no trained embedding -- it didn't appear "
                "in fit()'s `dataset1`/`dataset2`."
            )
        return vector
