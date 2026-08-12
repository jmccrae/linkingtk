"""Dataset loaders and Hugging Face integrations.

Loader implementations for individual datasets (DBP15K, OpenEA, ICEWS,
WordNet-Wikidata, AIDA-CoNLL, TAC KBP, ZESHEL, DaMuEL, LCQuAD 2.0, UFSAC,
SemCor, ...) live in dedicated modules under this package. See DESIGN.md's
Datasets section for the full list and sources.
"""

from linkingtk.datasets.base import DatasetLoader, GraphDatasetLoader
from linkingtk.datasets.dbp15k import DBP15KFrEnDataset, DBP15KJaEnDataset, DBP15KZhEnDataset
from linkingtk.datasets.icews import IcewsWikiDataset
from linkingtk.datasets.naisc import AnatomyDataset, ConferenceDataset
from linkingtk.datasets.openea import (
    DbpediaWikidata15KDataset,
    DbpediaYago15KDataset,
    EnDe15KDataset,
    EnFr15KDataset,
)
from linkingtk.datasets.openea_native import (
    DbpediaWikidata15KAttrDataset,
    DbpediaYago15KAttrDataset,
    EnDe15KAttrDataset,
    EnFr15KAttrDataset,
)
from linkingtk.datasets.toy import ToyEADataset, ToyELDataset, ToyWSADataset, ToyWSDDataset
from linkingtk.datasets.wordnet_wikidata import (
    WordNetWikidataLanguagesDataset,
    WordNetWikidataLocationsDataset,
    WordNetWikidataOrganismsDataset,
    WordNetWikidataOrganismsHardDataset,
)

__all__ = [
    "DatasetLoader",
    "GraphDatasetLoader",
    "ConferenceDataset",
    "AnatomyDataset",
    "ToyEADataset",
    "ToyWSDDataset",
    "ToyELDataset",
    "ToyWSADataset",
    "DBP15KZhEnDataset",
    "DBP15KJaEnDataset",
    "DBP15KFrEnDataset",
    "EnFr15KDataset",
    "EnDe15KDataset",
    "DbpediaWikidata15KDataset",
    "DbpediaYago15KDataset",
    "EnFr15KAttrDataset",
    "EnDe15KAttrDataset",
    "DbpediaWikidata15KAttrDataset",
    "DbpediaYago15KAttrDataset",
    "IcewsWikiDataset",
    "WordNetWikidataLanguagesDataset",
    "WordNetWikidataLocationsDataset",
    "WordNetWikidataOrganismsDataset",
    "WordNetWikidataOrganismsHardDataset",
]
