"""Loaders for OpenEA's sampled multilingual and homogeneous EA benchmarks.

Source: rehosted (see ``linkingtk.datasets.kg_zip`` for why) at
https://github.com/DexterZeng/EntMatcher's ``data.zip``, under
``data/{en_fr_15k_V1,en_de_15k_V1,dbp_wd_15k_V1,dbp_yg_15k_V1}/``.

Only the 15K/V1 size is available from this mirror -- OpenEA's
name-bias-fixed V2 release and the EN-FR/EN-DE 100K sizes are Figshare-only
(no stable fetchable URL, see ``linkingtk.datasets.kg_zip``'s module
docstring); D-W/D-Y do have a 100K variant in this same zip
(``dbp_wd_100``/``dbp_yg_100``) but it's not exposed here to keep the
multilingual and homogeneous sets available at consistent sizes.
"""

from __future__ import annotations

from linkingtk.datasets.kg_zip import _KGZipDataset

_ZIP_URL = "https://raw.githubusercontent.com/DexterZeng/EntMatcher/master/data.zip"


class EnFr15KDataset(_KGZipDataset):
    """OpenEA's EN-FR-15K-V1: English-French DBpedia (multilingual, ~15K/side)."""

    _zip_url = _ZIP_URL
    _folder = "data/en_fr_15k_V1"
    _ground_truth_files = ("ill_ent_ids",)
    _train_ground_truth_file = "sup_ent_ids"
    _test_ground_truth_file = "ref_ent_ids"
    _val_ground_truth_file = "val_ent_ids"


class EnDe15KDataset(_KGZipDataset):
    """OpenEA's EN-DE-15K-V1: English-German DBpedia (multilingual, ~15K/side)."""

    _zip_url = _ZIP_URL
    _folder = "data/en_de_15k_V1"
    _ground_truth_files = ("ill_ent_ids",)
    _train_ground_truth_file = "sup_ent_ids"
    _test_ground_truth_file = "ref_ent_ids"
    _val_ground_truth_file = "val_ent_ids"


class DbpediaWikidata15KDataset(_KGZipDataset):
    """OpenEA's D-W-15K-V1: DBpedia-Wikidata (homogeneous, ~15K/side)."""

    _zip_url = _ZIP_URL
    _folder = "data/dbp_wd_15k_V1"
    _ground_truth_files = ("ill_ent_ids",)
    _train_ground_truth_file = "sup_ent_ids"
    _test_ground_truth_file = "ref_ent_ids"
    _val_ground_truth_file = "val_ent_ids"


class DbpediaYago15KDataset(_KGZipDataset):
    """OpenEA's D-Y-15K-V1: DBpedia-YAGO (homogeneous, ~15K/side)."""

    _zip_url = _ZIP_URL
    _folder = "data/dbp_yg_15k_V1"
    _ground_truth_files = ("ill_ent_ids",)
    _train_ground_truth_file = "sup_ent_ids"
    _test_ground_truth_file = "ref_ent_ids"
    _val_ground_truth_file = "val_ent_ids"
