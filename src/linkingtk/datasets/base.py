"""Abstract interface for dataset loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod

from linkingtk.core.entity import Entity


class DatasetLoader(ABC):
    """Loads a linking dataset as a pair of entity lists plus ground truth.

    Concrete loaders fetch data from its original source where possible,
    falling back to a republished copy on the Hugging Face Hub (see
    DESIGN.md's Datasets section for the per-task dataset list).
    """

    @abstractmethod
    def load(self) -> tuple[list[Entity], list[Entity], list[tuple[str, str]]]:
        """Load the dataset.

        Returns:
            A tuple of ``(dataset1, dataset2, ground_truth)`` where
            ``ground_truth`` is a list of ``(source_id, target_id)`` pairs.
        """
