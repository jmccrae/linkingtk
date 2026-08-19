"""Zeshel (Zero-shot Entity Linking) dataset loader.

Zeshel (Logeswaran et al., "Zero-Shot Entity Linking by Reading Entity
Descriptions", 2019, https://github.com/lajanugen/zeshel) links named
mentions inside a Wikia (fandom) page's text to *other* Wikia pages, in
domains ("worlds") disjoint between train/validation/test -- entities the
model must resolve from description text alone, never seen at train
time. This is BLINK's own "Zero-shot EL" benchmark (Wu et al. 2020,
*Scalable Zero-shot Entity Linking with Dense Entity Retrieval*, Table
1): its bi-encoder-only Recall@64 numbers (93.12/91.44/82.06 train/val/
test) are the closest thing BLINK's own paper has to an apples-to-apples
bi-encoder target -- unlike TAC-KBP2010 (LDC-licensed, not freely
downloadable) or WikilinksNED Unseen-Mentions (no stable public host).
See [BlinkLinker][linkingtk.algorithms.el.blink.BlinkLinker].

The original release is Google-Drive-hosted (not automatable); this
loader instead uses a community republish on the Hugging Face Hub,
``naist-nlp/zeshel`` (mirrors the original CC-BY-SA Wikia content) via
the ``datasets`` library (already a hard dependency -- no new install).
Its ``train``/``validation``/``test`` splits already encode the paper's
zero-shot domain-disjoint protocol (each split's ``subset`` values are
disjoint domains). Split sizes don't exactly match the paper's original
mention counts (the original's exact split was never republished in an
automatable form), so treat any published-number comparison as
in-spirit, not a byte-identical reproduction.

Each mention is a hyperlink inside some entity's own Wikia page text,
pointing at a *different* entity -- so mentions and the KB come from two
separate configs of the same release: ``"dataset"`` (annotated documents)
and ``"dictionary"`` (each domain's own entity KB, name + description).
"""

from __future__ import annotations

from pathlib import Path

from linkingtk.core.entity import Entity
from linkingtk.datasets.base import DatasetLoader

_SOURCE = "naist-nlp/zeshel"
_MENTIONS_CONFIG = "dataset"
_KB_CONFIG = "dictionary"
_SPLITS = ("train", "validation", "test")


def _mention_id(document_id: str, start: int, end: int) -> str:
    return f"mention:{document_id}:{start}:{end}"


def _config_name(source: str, config_name: str) -> str | None:
    """The HF Hub repo's real config name, or ``None`` for a local directory.

    A local directory of parquet files (as tests use, to avoid a network
    fetch) has exactly one anonymous ``"default"`` config -- passing this
    release's real multi-config names (``"dataset"``/``"dictionary"``)
    would raise ``BuilderConfig '...' not found``.
    """
    return None if Path(source).is_dir() else config_name


def _kb_split_name(source: str) -> str:
    """The KB config's split name -- ``"kb"`` on the real release, ``"train"``
    for a local single-file test fixture (a local directory's lone parquet
    file is always named the ``"train"`` split, regardless of its filename)."""
    return "train" if Path(source).is_dir() else "kb"


class ZeshelDataset(DatasetLoader):
    """Loads Zeshel's zero-shot EL mentions and per-domain entity KB.

    Args:
        mentions_source: Hugging Face Hub id (or local directory, for
            tests) for the annotated-documents config.
        kb_source: Hugging Face Hub id (or local directory, for tests)
            for the entity-dictionary config.
    """

    def __init__(self, mentions_source: str = _SOURCE, kb_source: str = _SOURCE) -> None:
        self.mentions_source = mentions_source
        self.kb_source = kb_source

    def load(self) -> tuple[list[Entity], list[Entity], list[tuple[str, str]]]:
        mentions_by_split, ground_truth_by_split, domains = self._build_mentions(_SPLITS)
        mentions = [m for split in _SPLITS for m in mentions_by_split[split]]
        ground_truth = [gt for split in _SPLITS for gt in ground_truth_by_split[split]]
        kb = self._load_kb(domains)
        return mentions, kb, ground_truth

    def load_splits(
        self,
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
        """Load this dataset's native train/test/validation ground-truth split.

        Unlike ``load()``, this skips the (~1.2GB) entity-dictionary
        download entirely -- only mention/entity ids are needed. Same
        shape and argument order as
        [AidaConllDataset.load_splits][linkingtk.datasets.aida_conll.AidaConllDataset.load_splits].

        Returns:
            ``(train_pairs, test_pairs, val_pairs)``.
        """
        _mentions_by_split, ground_truth_by_split, _domains = self._build_mentions(_SPLITS)
        return (
            ground_truth_by_split["train"],
            ground_truth_by_split["test"],
            ground_truth_by_split["validation"],
        )

    def _build_mentions(
        self, splits: tuple[str, ...]
    ) -> tuple[dict[str, list[Entity]], dict[str, list[tuple[str, str]]], set[str]]:
        import datasets as hf_datasets

        raw = hf_datasets.load_dataset(
            self.mentions_source, name=_config_name(self.mentions_source, _MENTIONS_CONFIG)
        )

        mentions_by_split: dict[str, list[Entity]] = {}
        ground_truth_by_split: dict[str, list[tuple[str, str]]] = {}
        domains: set[str] = set()

        for split in splits:
            split_mentions: list[Entity] = []
            split_ground_truth: list[tuple[str, str]] = []
            for row in raw[split]:
                document_id = row["id"]
                text = row["text"]
                domain = row["subset"]
                domains.add(domain)
                for mention in row["entities"]:
                    if not mention["label"]:
                        continue
                    start, end = mention["start"], mention["end"]
                    mention_id = _mention_id(document_id, start, end)
                    target_id = mention["label"][0]
                    split_mentions.append(
                        Entity(
                            id=mention_id,
                            labels=[text[start:end]],
                            context=(text, start, end),
                            properties={"domain": domain},
                        )
                    )
                    split_ground_truth.append((mention_id, target_id))
            mentions_by_split[split] = split_mentions
            ground_truth_by_split[split] = split_ground_truth

        return mentions_by_split, ground_truth_by_split, domains

    def _load_kb(self, domains: set[str]) -> list[Entity]:
        import datasets as hf_datasets

        raw = hf_datasets.load_dataset(
            self.kb_source,
            name=_config_name(self.kb_source, _KB_CONFIG),
            split=_kb_split_name(self.kb_source),
        )
        raw = raw.filter(lambda row: row["subset"] in domains)
        return [
            Entity(
                id=row["id"],
                labels=[row["name"]],
                description=row["description"],
                properties={"domain": row["subset"]},
            )
            for row in raw
        ]
