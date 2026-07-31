"""Dataset loaders and Hugging Face integrations.

Loader implementations for individual datasets (DBP15K, OpenEA, ICEWS,
WordNet-Wikidata, AIDA-CoNLL, TAC KBP, ZESHEL, DaMuEL, LCQuAD 2.0, UFSAC,
SemCor, ...) live in dedicated modules under this package. See DESIGN.md's
Datasets section for the full list and sources.
"""

from linkingtk.datasets.base import DatasetLoader
from linkingtk.datasets.naisc import AnatomyDataset, ConferenceDataset
from linkingtk.datasets.toy import ToyEADataset, ToyELDataset, ToyWSADataset, ToyWSDDataset

__all__ = [
    "DatasetLoader",
    "ConferenceDataset",
    "AnatomyDataset",
    "ToyEADataset",
    "ToyWSDDataset",
    "ToyELDataset",
    "ToyWSADataset",
]
