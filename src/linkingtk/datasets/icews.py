"""Loader for ICEWS-Wiki, a heterogeneous event-KG-to-Wikipedia EA benchmark.

Source: https://github.com/jxh4945777/Simple-HHEA's
``data/icews_wiki.zip`` -- same ``ent_ids``/``triples`` format as
[linkingtk.datasets.dbp15k][]/[linkingtk.datasets.openea][]
(``triples_N`` here has two extra trailing timestamp-id columns, ignored by
the shared parser's ``load_graphs()``; see
[load_temporal_graphs][linkingtk.datasets.icews.IcewsWikiDataset.load_temporal_graphs]
for a loader that keeps them).

``icews_yago`` (the other dataset in that repo) is intentionally not
implemented: it's split across a 3-part archive
(``icews_yago.z01``/``.z02``/``.zip``) rather than a single zip, which
would need real multi-part-archive reassembly rather than the single
``fetch_cached()`` + `zipfile` this module and its siblings share.
"""

from __future__ import annotations

import re
import zipfile

from linkingtk.datasets.kg_zip import _KGZipDataset

TemporalTriple = tuple[str, str, str, str | None, str | None]

_TIME_LABEL_RE = re.compile(r"^\d+-\d+$")


class IcewsWikiDataset(_KGZipDataset):
    """ICEWS (event KG) aligned to Wikipedia entries."""

    _zip_url = "https://raw.githubusercontent.com/jxh4945777/Simple-HHEA/main/data/icews_wiki.zip"
    _folder = "icews_wiki"
    _ground_truth_files = ("ref_pairs", "sup_pairs")
    _train_ground_truth_file = "sup_pairs"
    _test_ground_truth_file = "ref_pairs"

    def _time_labels(self, archive: zipfile.ZipFile) -> dict[str, str | None]:
        """Local ``time_id`` -> a ``"YYYY-MM"`` label, or ``None`` if unresolvable.

        A handful of ``time_id`` entries are a bare (sometimes negative,
        e.g. ``-400000`` for a prehistoric Wikipedia date) year with no
        month -- unusable for
        [SimpleHHEALinker][linkingtk.algorithms.ea.simple_hhea.SimpleHHEALinker]'s
        month-bin histogram, so mapped to ``None`` rather than raising.
        """
        labels: dict[str, str | None] = {}
        for line in self._member(archive, "time_id").splitlines():
            local_id, raw = line.split("\t", 1)
            labels[local_id] = raw if _TIME_LABEL_RE.match(raw) else None
        return labels

    def _temporal_triples(
        self, archive: zipfile.ZipFile, side: int, time_labels: dict[str, str | None]
    ) -> list[TemporalTriple]:
        prefix = self._id_prefix(side)
        triples = []
        for line in self._member(archive, f"triples_{side}").splitlines():
            subject_id, predicate_id, object_id, start_id, end_id = line.split()
            triples.append(
                (
                    f"{prefix}{subject_id}",
                    predicate_id,
                    f"{prefix}{object_id}",
                    time_labels.get(start_id),
                    time_labels.get(end_id),
                )
            )
        return triples

    def load_temporal_graphs(self) -> tuple[list[TemporalTriple], list[TemporalTriple]]:
        """Like [load_graphs][linkingtk.datasets.base.GraphDatasetLoader.load_graphs],
        but keeps each triple's two timestamp columns instead of dropping them.

        Returns:
            `(subject_id, relation_id, object_id, start_label, end_label)`
            tuples per side, `start_label`/`end_label` a `"YYYY-MM"`
            string or `None` (unresolvable/missing timestamp) -- feed
            directly to
            [SimpleHHEALinker.fit][linkingtk.algorithms.ea.simple_hhea.SimpleHHEALinker.fit]'s
            `temporal_triples`.
        """
        with self._open_zip() as archive:
            time_labels = self._time_labels(archive)
            return (
                self._temporal_triples(archive, 1, time_labels),
                self._temporal_triples(archive, 2, time_labels),
            )
