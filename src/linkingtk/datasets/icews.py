"""Loader for ICEWS-Wiki, a heterogeneous event-KG-to-Wikipedia EA benchmark.

Source: https://github.com/jxh4945777/Simple-HHEA's
``data/icews_wiki.zip`` -- same ``ent_ids``/``triples`` format as
:mod:`~linkingtk.datasets.dbp15k`/:mod:`~linkingtk.datasets.openea`
(``triples_N`` here has two extra trailing timestamp-id columns, ignored by
the shared parser).

``icews_yago`` (the other dataset in that repo) is intentionally not
implemented: it's split across a 3-part archive
(``icews_yago.z01``/``.z02``/``.zip``) rather than a single zip, which
would need real multi-part-archive reassembly rather than the single
``fetch_cached()`` + `zipfile` this module and its siblings share.
"""

from __future__ import annotations

from linkingtk.datasets.kg_zip import _KGZipDataset


class IcewsWikiDataset(_KGZipDataset):
    """ICEWS (event KG) aligned to Wikipedia entries."""

    _zip_url = "https://raw.githubusercontent.com/jxh4945777/Simple-HHEA/main/data/icews_wiki.zip"
    _folder = "icews_wiki"
    _ground_truth_files = ("ref_pairs", "sup_pairs")
