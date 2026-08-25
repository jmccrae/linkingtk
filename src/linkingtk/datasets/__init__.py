"""Dataset loaders and Hugging Face integrations.

Loader implementations for individual datasets (DBP15K, OpenEA, ICEWS,
WordNet-Wikidata, AIDA-CoNLL, TAC KBP, ZESHEL, DaMuEL, LCQuAD 2.0, UFSAC,
SemCor, ...) live in dedicated modules under this package. See DESIGN.md's
Datasets section for the full list and sources.
"""

from linkingtk.datasets.aida_conll import AidaConllDataset
from linkingtk.datasets.base import DatasetLoader, GraphDatasetLoader
from linkingtk.datasets.damuel import DamuelDataset
from linkingtk.datasets.dbp15k import DBP15KFrEnDataset, DBP15KJaEnDataset, DBP15KZhEnDataset
from linkingtk.datasets.icews import IcewsWikiDataset
from linkingtk.datasets.lcquad2 import Lcquad2Dataset
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
from linkingtk.datasets.semcor import SemCorDataset
from linkingtk.datasets.toy import ToyEADataset, ToyELDataset, ToyWSADataset, ToyWSDDataset
from linkingtk.datasets.ufsac import UfsacDataset
from linkingtk.datasets.wikification import (
    Ace2004Dataset,
    AquaintDataset,
    MsnbcDataset,
    WikipediaSampleDataset,
)
from linkingtk.datasets.wordnet_wikidata import (
    WordNetWikidataLanguagesDataset,
    WordNetWikidataLocationsDataset,
    WordNetWikidataOrganismsDataset,
    WordNetWikidataOrganismsHardDataset,
)
from linkingtk.datasets.zeshel import ZeshelDataset

__all__ = [
    "DatasetLoader",
    "GraphDatasetLoader",
    "AidaConllDataset",
    "DamuelDataset",
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
    "Lcquad2Dataset",
    "WordNetWikidataLanguagesDataset",
    "WordNetWikidataLocationsDataset",
    "WordNetWikidataOrganismsDataset",
    "WordNetWikidataOrganismsHardDataset",
    "SemCorDataset",
    "UfsacDataset",
    "ZeshelDataset",
    "MsnbcDataset",
    "Ace2004Dataset",
    "AquaintDataset",
    "WikipediaSampleDataset",
]
