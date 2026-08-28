"""Entity Alignment (EA) algorithms.

EA finds matches between two knowledge graphs, often across languages.
Entities typically carry labels and descriptions but rarely a context, and
a supporting graph is normally available. Related systems: OpenEA,
Simple-HHEA, ChatEA, ProLEA, EntMatcher.
"""

from linkingtk.algorithms.ea.alinet import AliNetLinker
from linkingtk.algorithms.ea.attre import AttrELinker
from linkingtk.algorithms.ea.bootea import BootEALinker
from linkingtk.algorithms.ea.chatea import ChatEALinker
from linkingtk.algorithms.ea.entmatcher import EntMatcherLinker
from linkingtk.algorithms.ea.gcn_align import GCNAlignLinker
from linkingtk.algorithms.ea.imuse import IMUSELinker
from linkingtk.algorithms.ea.iptranse import IPTransELinker
from linkingtk.algorithms.ea.jape import JAPELinker
from linkingtk.algorithms.ea.kdcoe import KDCoELinker
from linkingtk.algorithms.ea.kge import KGELinker
from linkingtk.algorithms.ea.mtranse import MTransELinker
from linkingtk.algorithms.ea.multike import MultiKELinker
from linkingtk.algorithms.ea.rdgcn import RDGCNLinker
from linkingtk.algorithms.ea.rsn4ea import RSN4EALinker
from linkingtk.algorithms.ea.sea import SEALinker
from linkingtk.algorithms.ea.simple_hhea import SimpleHHEALinker

__all__ = [
    "AliNetLinker",
    "AttrELinker",
    "BootEALinker",
    "ChatEALinker",
    "EntMatcherLinker",
    "GCNAlignLinker",
    "IMUSELinker",
    "IPTransELinker",
    "JAPELinker",
    "KDCoELinker",
    "KGELinker",
    "MTransELinker",
    "MultiKELinker",
    "RDGCNLinker",
    "RSN4EALinker",
    "SEALinker",
    "SimpleHHEALinker",
]
