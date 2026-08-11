"""Loaders for DBP15K, the classic cross-lingual DBpedia EA benchmark.

Source: rehosted (see :mod:`~linkingtk.datasets.kg_zip` for why) at
https://github.com/DexterZeng/EntMatcher's ``data.zip``, under
``data/{zh_en,ja_en,fr_en}/``.
"""

from __future__ import annotations

from linkingtk.datasets.kg_zip import _KGZipDataset

_ZIP_URL = "https://raw.githubusercontent.com/DexterZeng/EntMatcher/master/data.zip"


class DBP15KZhEnDataset(_KGZipDataset):
    """DBP15K's Chinese-English pair (~15K entities per side)."""

    _zip_url = _ZIP_URL
    _folder = "data/zh_en"
    _ground_truth_files = ("ill_ent_ids",)


class DBP15KJaEnDataset(_KGZipDataset):
    """DBP15K's Japanese-English pair (~15K entities per side)."""

    _zip_url = _ZIP_URL
    _folder = "data/ja_en"
    _ground_truth_files = ("ill_ent_ids",)


class DBP15KFrEnDataset(_KGZipDataset):
    """DBP15K's French-English pair (~15K entities per side)."""

    _zip_url = _ZIP_URL
    _folder = "data/fr_en"
    _ground_truth_files = ("ill_ent_ids",)
