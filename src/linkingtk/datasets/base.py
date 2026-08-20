"""Abstract interface for dataset loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod

from linkingtk.core.entity import Entity
from linkingtk.core.source import EntitySource
from linkingtk.utils.graph import Graph


class DatasetLoader(ABC):
    """Loads a linking dataset as a pair of entity lists plus ground truth.

    Concrete loaders fetch data from its original source where possible,
    falling back to a republished copy on the Hugging Face Hub (see
    DESIGN.md's Datasets section for the per-task dataset list).
    """

    @abstractmethod
    def load(self) -> tuple[list[Entity], list[Entity] | EntitySource, list[tuple[str, str]]]:
        """Load the dataset.

        Returns:
            A tuple of ``(dataset1, dataset2, ground_truth)`` where
            ``ground_truth`` is a list of ``(source_id, target_id)`` pairs.
            ``dataset2`` is an [EntitySource][linkingtk.core.source.EntitySource]
            rather than a materialized list for loaders whose targets have
            no fixed candidate list to enumerate (e.g. a WSD corpus tagged
            against a full dictionary) -- see
            [SemCorDataset][linkingtk.datasets.semcor.SemCorDataset] and
            [UfsacDataset][linkingtk.datasets.ufsac.UfsacDataset].
        """


class GraphDatasetLoader(DatasetLoader):
    """A [DatasetLoader][linkingtk.datasets.base.DatasetLoader] whose datasets carry a
    supporting relational graph.

    Implemented by the graph-heavy real-world EA dataset loaders (DBP15K,
    OpenEA, ICEWS, WordNet-Wikidata) that back KGE-based linking -- unlike
    plain ``DatasetLoader``, ``load()`` alone isn't enough for those,
    since the relational structure (not just entity labels) is the point.
    """

    @abstractmethod
    def load_graphs(self) -> tuple[Graph, Graph]:
        """Load each side's supporting relational graph.

        Returns:
            ``(graph1, graph2)``, each in the same shape accepted by
            [link][linkingtk.algorithms.base.BaseLinker.link]'s ``graph``
            parameter -- here always a ``list[Triple]`` of
            ``(subject_id, predicate_id, object_id)``.
        """
