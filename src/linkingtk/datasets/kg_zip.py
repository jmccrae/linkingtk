"""Shared base for datasets hosted as a zip of numeric-id KG files.

Neither DBP15K's canonical host (Google Drive, via the JAPE paper's site)
nor OpenEA's (Dropbox/Figshare) is a stable, programmatically fetchable
URL. Two GitHub repos rehost the same data as plain zips in an identical
format instead -- ``github.com/DexterZeng/EntMatcher`` (DBP15K plus
OpenEA's EN-FR/EN-DE/D-W/D-Y-15K sets) and
``github.com/jxh4945777/Simple-HHEA`` (ICEWS) -- which is what the
concrete loaders in [linkingtk.datasets.dbp15k][],
[linkingtk.datasets.openea][] and [linkingtk.datasets.icews][]
point at.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import ClassVar

from linkingtk.core.entity import Entity
from linkingtk.datasets._util import fetch_cached, label_from_raw
from linkingtk.datasets.base import GraphDatasetLoader
from linkingtk.utils.graph import Graph, Triple


class _KGZipDataset(GraphDatasetLoader):
    """Base loader for the ``ent_ids``/``triples`` zip format.

    Within the zip, a dataset lives under ``_folder`` with per-side
    ``ent_ids_N`` (``id<TAB>uri_or_label``) and ``triples_N`` (whitespace-
    separated columns; the first three are always ``subject_id
    predicate_id object_id`` -- extra trailing columns, as ICEWS has for
    event timestamps, are ignored) files, plus one or more
    ``_ground_truth_files`` listing known-correct ``id1<TAB>id2`` alignment
    pairs (concatenated if there's more than one, e.g. separate train/test
    files with no single combined file).

    Some datasets additionally ship a native supervised train/test/
    validation split of that same ground truth (as separate files) --
    where ``_train_ground_truth_file``/``_test_ground_truth_file`` are set,
    [load_splits][linkingtk.datasets.kg_zip._KGZipDataset.load_splits]
    exposes that split directly, instead of ``load()``'s single
    concatenated list.
    """

    _zip_url: ClassVar[str]
    _folder: ClassVar[str]
    _ground_truth_files: ClassVar[tuple[str, ...]]
    _train_ground_truth_file: ClassVar[str]
    _test_ground_truth_file: ClassVar[str]
    _val_ground_truth_file: ClassVar[str | None] = None

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

    @property
    def _dataset_name(self) -> str:
        return self._folder.rsplit("/", 1)[-1]

    def _id_prefix(self, side: int) -> str:
        return f"{self._dataset_name}:{side}:"

    def _open_zip(self) -> zipfile.ZipFile:
        return zipfile.ZipFile(io.BytesIO(fetch_cached(self.zip_url, self.cache_dir)))

    def _member(self, archive: zipfile.ZipFile, name: str) -> str:
        return archive.read(f"{self._folder}/{name}").decode("utf-8")

    def _entities(self, archive: zipfile.ZipFile, side: int) -> tuple[list[Entity], dict[str, str]]:
        prefix = self._id_prefix(side)
        ids = {}
        entities = []
        for line in self._member(archive, f"ent_ids_{side}").splitlines():
            local_id, raw = line.split("\t", 1)
            entity_id = f"{prefix}{local_id}"
            ids[local_id] = entity_id
            entities.append(Entity(id=entity_id, labels=[label_from_raw(raw)]))
        return entities, ids

    def _pairs_from_file(
        self,
        archive: zipfile.ZipFile,
        ids1: dict[str, str],
        ids2: dict[str, str],
        filename: str,
    ) -> list[tuple[str, str]]:
        pairs = []
        for line in self._member(archive, filename).splitlines():
            id1, id2 = line.split("\t")
            if id1 in ids1 and id2 in ids2:
                pairs.append((ids1[id1], ids2[id2]))
        return pairs

    def load(self) -> tuple[list[Entity], list[Entity], list[tuple[str, str]]]:
        with self._open_zip() as archive:
            entities1, ids1 = self._entities(archive, 1)
            entities2, ids2 = self._entities(archive, 2)
            ground_truth = [
                pair
                for filename in self._ground_truth_files
                for pair in self._pairs_from_file(archive, ids1, ids2, filename)
            ]
        return entities1, entities2, ground_truth

    def load_splits(
        self,
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
        """Load this dataset's native train/test/validation ground-truth split.

        Unlike ``load()``'s single
        concatenated ``ground_truth``, this preserves the dataset's own
        supervised split -- the train pairs are meant to be added to the
        training graph as seed alignment triples (see
        [KGELinker.fit][linkingtk.algorithms.ea.kge.KGELinker.fit]), with
        the test (and, where available, validation) pairs held out for
        ranked evaluation.

        Returns:
            ``(train_pairs, test_pairs, val_pairs)``. ``val_pairs`` is
            ``[]`` for datasets with no native validation split.
        """
        with self._open_zip() as archive:
            _, ids1 = self._entities(archive, 1)
            _, ids2 = self._entities(archive, 2)
            train = self._pairs_from_file(archive, ids1, ids2, self._train_ground_truth_file)
            test = self._pairs_from_file(archive, ids1, ids2, self._test_ground_truth_file)
            val = (
                self._pairs_from_file(archive, ids1, ids2, self._val_ground_truth_file)
                if self._val_ground_truth_file is not None
                else []
            )
        return train, test, val

    def _triples(self, archive: zipfile.ZipFile, side: int) -> list[Triple]:
        prefix = self._id_prefix(side)
        triples = []
        for line in self._member(archive, f"triples_{side}").splitlines():
            subject_id, predicate_id, object_id = line.split()[:3]
            triples.append((f"{prefix}{subject_id}", predicate_id, f"{prefix}{object_id}"))
        return triples

    def load_graphs(self) -> tuple[Graph, Graph]:
        with self._open_zip() as archive:
            return self._triples(archive, 1), self._triples(archive, 2)
