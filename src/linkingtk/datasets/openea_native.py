"""Loaders for OpenEA's own native per-triple format, including attribute triples.

Source: rehosted on the Hugging Face Hub by the matchbench project
(https://huggingface.co/matchbench) as plain zips of OpenEA's own raw-URI
``rel_triples_N``/``attr_triples_N``/``train_links``/``valid_links``/
``test_links`` files -- unlike [linkingtk.datasets.openea][]'s EntMatcher-
rehosted, numeric-id, structural-only loaders, this format ships attribute
triples too, needed by EA methods that use attribute signal (JAPE #28, and
later AttrE/IMUSE/MultiKE).

**Not a drop-in replacement** for [linkingtk.datasets.openea][]'s loaders:
cross-checked while investigating #28 by downloading both zips and comparing
entity URI sets -- only ~10% of `EnFr15KDataset`'s (EntMatcher rehost)
entity URIs coincide with `EnFr15KAttrDataset`'s (this module's) entities.
They're two independently-sampled cuts of "EN-FR-15K" that happen to share
the same split-size convention (OpenEA's standard 20/10/70% ratio), not the
same entity roster. So these loaders are entirely self-contained (own
entity ids, own relation/attribute triples, own train/valid/test split, all
from one internally-consistent source) rather than joined onto
[linkingtk.datasets.openea][]'s loaders via URI matching, which would only
cover a small fraction of entities. Use this module specifically when
attribute triples are needed; use [linkingtk.datasets.openea][] otherwise
(already used/verified by MTransE #26, IPTransE #27).
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import ClassVar

from linkingtk.core.entity import Entity
from linkingtk.datasets._util import fetch_cached, label_from_raw, parse_literal_value
from linkingtk.datasets.base import GraphDatasetLoader
from linkingtk.utils.graph import Graph, Triple

_LINK_FILES = ("train_links", "valid_links", "test_links")


class _OpenEANativeDataset(GraphDatasetLoader):
    """Base loader for OpenEA's native per-triple format (raw URIs, with attributes).

    Entities are implicit (no dedicated id file, unlike
    [linkingtk.datasets.kg_zip][]'s ``ent_ids_N``): each side's roster is
    the sorted set of every subject/object URI appearing in that side's
    ``rel_triples_N``, enumerated into ``{prefix}{index}`` ids. Attribute
    triples or split pairs referencing a URI absent from ``rel_triples_N``
    are dropped (tolerated, same posture as
    [linkingtk.datasets.kg_zip][]'s dangling-reference handling). The zip
    has no folder prefix -- files live at its root.
    """

    _zip_url: ClassVar[str]
    _dataset_name: ClassVar[str]

    def __init__(self, zip_url: str | None = None, cache_dir: Path | None = None) -> None:
        """Create the loader.

        Args:
            zip_url: Override for where the dataset zip is fetched from (a
                URL or ``file://`` path). Defaults to this dataset's usual
                hosted archive.
            cache_dir: Override for the download cache directory. Ignored
                for ``file://`` URLs, which are never cached.
        """
        self.zip_url = zip_url if zip_url is not None else self._zip_url
        self.cache_dir = cache_dir

    def _id_prefix(self, side: int) -> str:
        return f"{self._dataset_name}:{side}:"

    def _open_zip(self) -> zipfile.ZipFile:
        return zipfile.ZipFile(io.BytesIO(fetch_cached(self.zip_url, self.cache_dir)))

    def _member(self, archive: zipfile.ZipFile, name: str) -> str:
        return archive.read(name).decode("utf-8")

    def _entity_ids(self, archive: zipfile.ZipFile, side: int) -> dict[str, str]:
        prefix = self._id_prefix(side)
        uris: set[str] = set()
        for line in self._member(archive, f"rel_triples_{side}").splitlines():
            subject, _, obj = line.split("\t")
            uris.add(subject)
            uris.add(obj)
        return {uri: f"{prefix}{index}" for index, uri in enumerate(sorted(uris))}

    def _entities(self, uri_to_id: dict[str, str]) -> list[Entity]:
        return [
            Entity(id=entity_id, labels=[label_from_raw(uri)])
            for uri, entity_id in uri_to_id.items()
        ]

    def _pairs_from_file(
        self,
        archive: zipfile.ZipFile,
        ids1: dict[str, str],
        ids2: dict[str, str],
        filename: str,
    ) -> list[tuple[str, str]]:
        pairs = []
        for line in self._member(archive, filename).splitlines():
            uri1, uri2 = line.split("\t")
            if uri1 in ids1 and uri2 in ids2:
                pairs.append((ids1[uri1], ids2[uri2]))
        return pairs

    def load(self) -> tuple[list[Entity], list[Entity], list[tuple[str, str]]]:
        with self._open_zip() as archive:
            ids1 = self._entity_ids(archive, 1)
            ids2 = self._entity_ids(archive, 2)
            entities1, entities2 = self._entities(ids1), self._entities(ids2)
            ground_truth = [
                pair
                for filename in _LINK_FILES
                for pair in self._pairs_from_file(archive, ids1, ids2, filename)
            ]
        return entities1, entities2, ground_truth

    def load_splits(
        self,
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
        """Load this dataset's native train/test/validation ground-truth split.

        Returns:
            ``(train_pairs, test_pairs, val_pairs)`` -- see
            [_KGZipDataset.load_splits][linkingtk.datasets.kg_zip._KGZipDataset.load_splits]
            for how these are meant to be used.
        """
        with self._open_zip() as archive:
            ids1, ids2 = self._entity_ids(archive, 1), self._entity_ids(archive, 2)
            train = self._pairs_from_file(archive, ids1, ids2, "train_links")
            test = self._pairs_from_file(archive, ids1, ids2, "test_links")
            val = self._pairs_from_file(archive, ids1, ids2, "valid_links")
        return train, test, val

    def _triples(self, archive: zipfile.ZipFile, side: int, ids: dict[str, str]) -> list[Triple]:
        triples = []
        for line in self._member(archive, f"rel_triples_{side}").splitlines():
            subject, predicate, obj = line.split("\t")
            triples.append((ids[subject], predicate, ids[obj]))
        return triples

    def load_graphs(self) -> tuple[Graph, Graph]:
        with self._open_zip() as archive:
            ids1, ids2 = self._entity_ids(archive, 1), self._entity_ids(archive, 2)
            return self._triples(archive, 1, ids1), self._triples(archive, 2, ids2)

    def _attr_triples(
        self, archive: zipfile.ZipFile, side: int, ids: dict[str, str]
    ) -> list[Triple]:
        triples = []
        for line in self._member(archive, f"attr_triples_{side}").splitlines():
            subject, predicate, value = line.split("\t", 2)
            if subject in ids:
                triples.append((ids[subject], predicate, parse_literal_value(value)))
        return triples

    def load_attribute_triples(self) -> tuple[list[Triple], list[Triple]]:
        """Load each side's ``(entity_id, predicate_uri, value)`` attribute triples.

        ``value`` has RDF literal quoting/``^^<datatype>``/``@lang``
        suffixes stripped via
        [parse_literal_value][linkingtk.datasets._util.parse_literal_value].
        Not part of the [GraphDatasetLoader][linkingtk.datasets.base.GraphDatasetLoader]
        interface -- only this loader family provides it so far; promote
        to a proper mixin/ABC once a second attribute-bearing loader
        family exists.

        Returns:
            ``(attribute_triples1, attribute_triples2)``. Triples whose
            subject URI doesn't appear in that side's ``rel_triples_N``
            are dropped.
        """
        with self._open_zip() as archive:
            ids1, ids2 = self._entity_ids(archive, 1), self._entity_ids(archive, 2)
            return self._attr_triples(archive, 1, ids1), self._attr_triples(archive, 2, ids2)


class EnFr15KAttrDataset(_OpenEANativeDataset):
    """OpenEA's EN-FR-15K-V1, native format with attributes (English-French DBpedia)."""

    _zip_url = (
        "https://huggingface.co/datasets/matchbench/openea-en-fr-15k-v1/"
        "resolve/main/openea-en-fr-15k-v1.zip"
    )
    _dataset_name = "en_fr_15k_v1_attr"


class EnDe15KAttrDataset(_OpenEANativeDataset):
    """OpenEA's EN-DE-15K-V1, native format with attributes (English-German DBpedia)."""

    _zip_url = (
        "https://huggingface.co/datasets/matchbench/openea-en-de-15k-v1/"
        "resolve/main/openea-en-de-15k-v1.zip"
    )
    _dataset_name = "en_de_15k_v1_attr"


class DbpediaWikidata15KAttrDataset(_OpenEANativeDataset):
    """OpenEA's D-W-15K-V1, native format with attributes (DBpedia-Wikidata)."""

    _zip_url = (
        "https://huggingface.co/datasets/matchbench/openea-d-w-15k-v1/"
        "resolve/main/openea-d-w-15k-v1.zip"
    )
    _dataset_name = "dbp_wd_15k_v1_attr"


class DbpediaYago15KAttrDataset(_OpenEANativeDataset):
    """OpenEA's D-Y-15K-V1, native format with attributes (DBpedia-YAGO)."""

    _zip_url = (
        "https://huggingface.co/datasets/matchbench/openea-d-y-15k-v1/"
        "resolve/main/openea-d-y-15k-v1.zip"
    )
    _dataset_name = "dbp_yg_15k_v1_attr"
