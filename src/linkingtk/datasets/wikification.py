"""MSNBC, ACE2004, AQUAINT, and WikipediaSample Entity Linking dataset loaders.

All four datasets are distributed together in one archive from the UPenn
CogComp group -- the data behind Ratinov et al. (2011)'s wikifier, which
also carries Cucerzan (2007)'s original MSNBC annotations and Milne &
Witten's AQUAINT/ACE2004 annotations:
``https://cogcomp.seas.upenn.edu/Data/ACL2011WikificationData.zip``.

Each document has a ``ReferenceProblem`` gold file (one ``ReferenceInstance``
per mention: ``SurfaceForm``, character ``Offset``/``Length``, and a
``ChosenAnnotation`` target -- either a full ``en.wikipedia.org/wiki/<Title>``
URL, as MSNBC/ACE2004/AQUAINT use, or a bare title, as WikipediaSample's
Wikipedia-internal links use) plus a matching raw document-text file that
``Offset``/``Length`` index into directly -- no NIF-style nested-context
offset arithmetic needed. ``ChosenAnnotation`` is the literal string
``"*null*"`` for unlinkable (NIL) mentions; these are skipped entirely,
matching [AidaConllDataset][linkingtk.datasets.aida_conll.AidaConllDataset]'s
"skip NIL, InKB-only" convention.

WikipediaSample's train split alone is 9938 documents (~266K mentions) of
auto-derived Wikipedia-internal-link annotations -- too much to parse and
description-fetch by default on every ``load()``, so
[WikipediaSampleDataset][linkingtk.datasets.wikification.WikipediaSampleDataset]
caps it via ``max_train_documents``, following the precedent set by
[DamuelDataset][linkingtk.datasets.damuel.DamuelDataset]'s ``max_parts`` for
capping a large auto-derived corpus by default.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import ClassVar

from linkingtk.core.entity import Entity
from linkingtk.datasets._util import fetch_cached, local_name
from linkingtk.datasets.aida_conll import DescriptionFetcher, fetch_wikipedia_extracts
from linkingtk.datasets.base import DatasetLoader

_ZIP_URL = "https://cogcomp.seas.upenn.edu/Data/ACL2011WikificationData.zip"
_NIL = "*null*"

_INSTANCE_RE = re.compile(r"<ReferenceInstance>(.*?)</ReferenceInstance>", re.DOTALL)


def _tag(block: str, name: str) -> str:
    match = re.search(rf"<{name}>\s*(.*?)\s*</{name}>", block, re.DOTALL)
    return match.group(1) if match else ""


def _entity_title(chosen_annotation: str) -> str:
    if chosen_annotation.startswith("http"):
        return local_name(chosen_annotation)
    return chosen_annotation


def _members(archive: zipfile.ZipFile, directory: str) -> list[str]:
    prefix = f"{directory}/"
    return sorted(
        name
        for name in archive.namelist()
        if name.startswith(prefix) and not name.endswith("/") and ".svn" not in name
    )


class _WikificationDataset(DatasetLoader):
    """Shared base for the ACL2011WikificationData ``Problems``/``RawTexts`` format."""

    _problems_dir: ClassVar[str]
    _raw_texts_dir: ClassVar[str]

    def __init__(
        self,
        zip_url: str | None = None,
        cache_dir: Path | None = None,
        description_fetcher: DescriptionFetcher | None = None,
    ) -> None:
        """Create the loader.

        Args:
            zip_url: Override for where the shared ``ACL2011WikificationData.zip``
                archive is fetched from (a URL or ``file://`` path).
            cache_dir: Override for the download cache directory. Ignored if
                ``description_fetcher`` is given.
            description_fetcher: Overrides how KB entity descriptions are
                sourced, given the list of unique Wikipedia titles this
                dataset needs. Defaults to
                [fetch_wikipedia_extracts][linkingtk.datasets.aida_conll.fetch_wikipedia_extracts]
                -- pass a fake here in tests to avoid a real network call.
        """
        self.zip_url = zip_url if zip_url is not None else _ZIP_URL
        self.cache_dir = cache_dir
        self.description_fetcher = (
            description_fetcher
            if description_fetcher is not None
            else lambda titles: fetch_wikipedia_extracts(titles, self.cache_dir)
        )

    def _open_zip(self) -> zipfile.ZipFile:
        return zipfile.ZipFile(io.BytesIO(fetch_cached(self.zip_url, self.cache_dir)))

    def _load_dir(
        self,
        archive: zipfile.ZipFile,
        problems_dir: str,
        raw_texts_dir: str,
        max_documents: int | None = None,
    ) -> tuple[list[Entity], list[tuple[str, str]], dict[str, str]]:
        """Parse every document under ``problems_dir``, matched against ``raw_texts_dir``.

        Returns:
            ``(mentions, ground_truth, titles_by_id)`` where ``titles_by_id``
            maps a KB entity id (the bare Wikipedia title) to itself -- kept
            as a dict (not a set) so it composes directly with
            [AidaConllDataset][linkingtk.datasets.aida_conll.AidaConllDataset]'s
            ``titles_by_id.values()`` convention for the description fetcher.
        """
        problem_names = _members(archive, problems_dir)
        if max_documents is not None:
            problem_names = problem_names[:max_documents]

        mentions: list[Entity] = []
        ground_truth: list[tuple[str, str]] = []
        titles_by_id: dict[str, str] = {}

        for problem_name in problem_names:
            doc_id = problem_name.rsplit("/", 1)[-1]
            raw_text = archive.read(f"{raw_texts_dir}/{doc_id}").decode("utf-8", errors="replace")
            problem_text = archive.read(problem_name).decode("utf-8", errors="replace")

            for block in _INSTANCE_RE.findall(problem_text):
                chosen_annotation = _tag(block, "ChosenAnnotation")
                if chosen_annotation == _NIL:
                    continue
                start = int(_tag(block, "Offset"))
                end = start + int(_tag(block, "Length"))
                mention_id = f"mention:{doc_id}:{start}:{end}"
                title = _entity_title(chosen_annotation)
                mentions.append(
                    Entity(
                        id=mention_id,
                        labels=[raw_text[start:end]],
                        context=(raw_text, start, end),
                    )
                )
                titles_by_id.setdefault(title, title)
                ground_truth.append((mention_id, title))

        return mentions, ground_truth, titles_by_id

    def _kb_entities(self, titles_by_id: dict[str, str], fetch_descriptions: bool) -> list[Entity]:
        descriptions = (
            self.description_fetcher(list(titles_by_id.values())) if fetch_descriptions else {}
        )
        return [
            Entity(
                id=entity_id,
                labels=[title.replace("_", " ")],
                description=descriptions.get(title, ""),
            )
            for entity_id, title in titles_by_id.items()
        ]


class MsnbcDataset(_WikificationDataset):
    """MSNBC: Cucerzan (2007)'s 20-document news-wire EL gold standard.

    ``dataset1`` (mentions) carries ``context=(text, start, end)``; ``dataset2``
    (KB entries) carries a label and a fetched Wikipedia lead-paragraph
    description, same shape as
    [AidaConllDataset][linkingtk.datasets.aida_conll.AidaConllDataset]. NIL
    mentions are skipped (5 of 567 in the full dataset).
    """

    _problems_dir = "WikificationACL2011Data/MSNBC/Problems"
    _raw_texts_dir = "WikificationACL2011Data/MSNBC/RawTexts"

    def load(self) -> tuple[list[Entity], list[Entity], list[tuple[str, str]]]:
        with self._open_zip() as archive:
            mentions, ground_truth, titles_by_id = self._load_dir(
                archive, self._problems_dir, self._raw_texts_dir
            )
            kb = self._kb_entities(titles_by_id, fetch_descriptions=True)
        return mentions, kb, ground_truth


class Ace2004Dataset(_WikificationDataset):
    """ACE2004: Milne & Witten's Turker-annotated ACE-coref-mention EL gold standard.

    Same shape as [MsnbcDataset][linkingtk.datasets.wikification.MsnbcDataset].
    NIL mentions are skipped (49 of 303 in the full dataset).
    """

    _problems_dir = "WikificationACL2011Data/ACE2004_Coref_Turking/Dev/ProblemsNoTranscripts"
    _raw_texts_dir = "WikificationACL2011Data/ACE2004_Coref_Turking/Dev/RawTextsNoTranscripts"

    def load(self) -> tuple[list[Entity], list[Entity], list[tuple[str, str]]]:
        with self._open_zip() as archive:
            mentions, ground_truth, titles_by_id = self._load_dir(
                archive, self._problems_dir, self._raw_texts_dir
            )
            kb = self._kb_entities(titles_by_id, fetch_descriptions=True)
        return mentions, kb, ground_truth


class AquaintDataset(_WikificationDataset):
    """AQUAINT: Milne & Witten's 50-document newswire EL gold standard.

    Same shape as [MsnbcDataset][linkingtk.datasets.wikification.MsnbcDataset].
    Has no NIL mentions (0 of 727 in the full dataset).
    """

    _problems_dir = "WikificationACL2011Data/AQUAINT/Problems"
    _raw_texts_dir = "WikificationACL2011Data/AQUAINT/RawTexts"

    def load(self) -> tuple[list[Entity], list[Entity], list[tuple[str, str]]]:
        with self._open_zip() as archive:
            mentions, ground_truth, titles_by_id = self._load_dir(
                archive, self._problems_dir, self._raw_texts_dir
            )
            kb = self._kb_entities(titles_by_id, fetch_descriptions=True)
        return mentions, kb, ground_truth


class WikipediaSampleDataset(_WikificationDataset):
    """WikipediaSample: Ratinov et al. (2011)'s auto-derived wikifier training/test set.

    Mentions come from Wikipedia articles' own internal wikilinks, not
    manual annotation -- so ``ChosenAnnotation`` is a bare title (e.g.
    ``New_Hampshire``), not a full URL, unlike
    [MsnbcDataset][linkingtk.datasets.wikification.MsnbcDataset] and siblings.
    The train split is 9938 documents (~266K mentions); ``max_train_documents``
    caps how many of those are parsed by default, following
    [DamuelDataset][linkingtk.datasets.damuel.DamuelDataset]'s ``max_parts``
    precedent for a large auto-derived corpus. The test split (40 documents)
    is always loaded in full.
    """

    _train_problems_dir = "WikificationACL2011Data/WikipediaSample/ProblemsTrain"
    _train_raw_texts_dir = "WikificationACL2011Data/WikipediaSample/RawTextsTrain"
    _test_problems_dir = "WikificationACL2011Data/WikipediaSample/ProblemsTest"
    _test_raw_texts_dir = "WikificationACL2011Data/WikipediaSample/RawTextsTest"

    def __init__(
        self,
        zip_url: str | None = None,
        cache_dir: Path | None = None,
        description_fetcher: DescriptionFetcher | None = None,
        max_train_documents: int | None = 500,
    ) -> None:
        """Create the loader.

        Args:
            zip_url: See
                [_WikificationDataset][linkingtk.datasets.wikification._WikificationDataset].
            cache_dir: See
                [_WikificationDataset][linkingtk.datasets.wikification._WikificationDataset].
            description_fetcher: See
                [_WikificationDataset][linkingtk.datasets.wikification._WikificationDataset].
            max_train_documents: How many of the train split's 9938
                documents to parse. ``None`` loads all of them (~266K
                mentions -- slow, and description-fetches every unique KB
                entity unless ``load_splits()`` is used instead).
        """
        super().__init__(
            zip_url=zip_url, cache_dir=cache_dir, description_fetcher=description_fetcher
        )
        self.max_train_documents = max_train_documents

    def load(self) -> tuple[list[Entity], list[Entity], list[tuple[str, str]]]:
        with self._open_zip() as archive:
            train_mentions, train_ground_truth, train_titles = self._load_dir(
                archive,
                self._train_problems_dir,
                self._train_raw_texts_dir,
                self.max_train_documents,
            )
            test_mentions, test_ground_truth, test_titles = self._load_dir(
                archive, self._test_problems_dir, self._test_raw_texts_dir
            )
            kb = self._kb_entities({**train_titles, **test_titles}, fetch_descriptions=True)
        return train_mentions + test_mentions, kb, train_ground_truth + test_ground_truth

    def load_splits(
        self,
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
        """Load this dataset's native train/test ground-truth split.

        Unlike ``load()``, this skips description fetching entirely (only
        mention/entity ids are needed) -- same shape and argument order as
        [AidaConllDataset.load_splits][linkingtk.datasets.aida_conll.AidaConllDataset.load_splits].

        Returns:
            ``(train_pairs, test_pairs, val_pairs)`` -- ``val_pairs`` is
            always ``[]``, this dataset has no native validation split.
        """
        with self._open_zip() as archive:
            _, train_ground_truth, _ = self._load_dir(
                archive,
                self._train_problems_dir,
                self._train_raw_texts_dir,
                self.max_train_documents,
            )
            _, test_ground_truth, _ = self._load_dir(
                archive, self._test_problems_dir, self._test_raw_texts_dir
            )
        return train_ground_truth, test_ground_truth, []
