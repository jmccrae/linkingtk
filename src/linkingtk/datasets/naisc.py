"""Loaders for Naisc's ontology-matching toy benchmarks.

Source: https://github.com/insight-centre/naisc/tree/master/datasets

Each dataset is a pair of small OWL ontologies (``left.rdf``/``right.rdf``)
plus a reference alignment (``align.rdf``) listing known-correct class
pairs between them. Both were originally used in the OAEI ontology
matching campaigns; only the classes (not object/data properties) are
loaded as entities here, which keeps both sides comfortably under 100
entities for :class:`ConferenceDataset` and lets
:meth:`Entity.description` stay unused (these ontologies rarely carry
``rdfs:comment``) in favor of just ``rdfs:label``.
"""

from __future__ import annotations

import re
from typing import ClassVar
from urllib.request import urlopen

from linkingtk.core.entity import Entity
from linkingtk.datasets.base import DatasetLoader
from linkingtk.exceptions import OptionalDependencyError

_BASE_URL = "https://raw.githubusercontent.com/insight-centre/naisc/master/datasets"

_ALIGN_LINE_RE = re.compile(r"<([^>]+)>\s+<[^>]+>\s+<([^>]+)>")


def _fetch(url: str) -> str:
    with urlopen(url) as response:
        data: bytes = response.read()
    return data.decode("utf-8")


def _local_name(uri: str) -> str:
    return uri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def _classes_from_ontology(data: str) -> list[Entity]:
    try:
        from rdflib import OWL, RDF, RDFS, Graph, URIRef
    except ImportError as exc:
        raise OptionalDependencyError("Naisc dataset loaders", "graph") from exc

    graph = Graph()
    graph.parse(data=data, format="xml")

    entities = []
    for subject in graph.subjects(RDF.type, OWL.Class):
        if not isinstance(subject, URIRef):
            continue
        label = graph.value(subject, RDFS.label)
        comment = graph.value(subject, RDFS.comment)
        entities.append(
            Entity(
                id=str(subject),
                labels=[str(label) if label else _local_name(str(subject)).replace("_", " ")],
                description=str(comment) if comment else None,
            )
        )
    return entities


def _ground_truth_from_alignment(
    data: str, left_ids: set[str], right_ids: set[str]
) -> list[tuple[str, str]]:
    pairs = []
    for line in data.splitlines():
        match = _ALIGN_LINE_RE.match(line.strip())
        if match and match.group(1) in left_ids and match.group(2) in right_ids:
            pairs.append((match.group(1), match.group(2)))
    return pairs


class _NaiscToyDataset(DatasetLoader):
    """Base loader for Naisc's two-ontology, single-alignment toy datasets."""

    _name: ClassVar[str]

    def __init__(self, base_url: str | None = None) -> None:
        """Create the loader.

        Args:
            base_url: Override for where ``left.rdf``/``right.rdf``/``align.rdf``
                are fetched from (a URL or ``file://`` path). Defaults to
                this dataset's directory in the Naisc GitHub repository.
        """
        self.base_url = base_url if base_url is not None else f"{_BASE_URL}/{self._name}"

    def load(self) -> tuple[list[Entity], list[Entity], list[tuple[str, str]]]:
        left = _classes_from_ontology(_fetch(f"{self.base_url}/left.rdf"))
        right = _classes_from_ontology(_fetch(f"{self.base_url}/right.rdf"))
        left_ids = {entity.id for entity in left}
        right_ids = {entity.id for entity in right}
        ground_truth = _ground_truth_from_alignment(
            _fetch(f"{self.base_url}/align.rdf"), left_ids, right_ids
        )
        return left, right, ground_truth


class ConferenceDataset(_NaiscToyDataset):
    """The OAEI 'conference' track: matching academic-conference ontologies.

    Well under 100 classes per side, making it suitable as a genuinely
    small Entity Alignment / Word Sense Alignment toy benchmark.
    """

    _name = "conference"


class AnatomyDataset(_NaiscToyDataset):
    """The OAEI 'anatomy' track: mouse-anatomy vs. NCI Thesaurus ontologies.

    Despite being listed alongside :class:`ConferenceDataset` as a toy
    dataset in Naisc, this is the full anatomy track (thousands of
    classes per side) — use :class:`ConferenceDataset` if you need a
    dataset small enough for quick, exhaustive testing.
    """

    _name = "anatomy"
