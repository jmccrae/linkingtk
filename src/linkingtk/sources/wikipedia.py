"""EntitySource wrapper for Wikipedia, via the public MediaWiki API.

Wraps the [MediaWiki API](https://www.mediawiki.org/wiki/API:Main_page) as a
query-driven [EntitySource][linkingtk.core.source.EntitySource], so EL can
target live Wikipedia directly -- without downloading or materializing a
dump -- the same way [WnEntitySource][linkingtk.sources.wn.WnEntitySource]
targets a local WordNet.
"""

from __future__ import annotations

import html
import re
from typing import TYPE_CHECKING

from linkingtk.core.entity import Entity
from linkingtk.core.source import EntitySource
from linkingtk.exceptions import OptionalDependencyError

if TYPE_CHECKING:
    import requests

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_DISAMBIGUATOR_RE = re.compile(r"^(?P<base>.+?)\s*\((?P<topic>[^()]+)\)$")

_USER_AGENT = "linkingtk/0.1 (https://github.com/jmccrae/linkingtk)"


def _clean_snippet(snippet: str) -> str:
    """Strip the MediaWiki search API's ``<span class="searchmatch">`` markup
    and HTML entities (e.g. ``&amp;``) out of a raw search result snippet.
    """
    return html.unescape(_HTML_TAG_RE.sub("", snippet))


def _split_title(title: str) -> tuple[str, str | None]:
    """Split a Wikipedia title into its bare label and disambiguating topic.

    Wikipedia disambiguates same-named articles via a parenthetical title
    suffix (e.g. ``"Paris (mythology)"``) rather than distinct labels the
    way WordNet's lemmas do -- so
    [ExactMatch][linkingtk.blocking.exact.ExactMatch]'s exact-label
    blocking would never treat a disambiguated page as a candidate for a
    bare-word mention (e.g. ``"Paris"``): confirmed directly,
    ``ExactMatch().candidate_pairs([Entity(id="m", labels=["Paris"])],
    WikipediaEntitySource())`` returns only the undisambiguated ``"Paris"``
    page, never ``"Paris (mythology)"``, even though `search` itself
    returns it. The label is stripped of the suffix so it can match; the
    topic moves to the description instead (via `_prefix_topic`) so it
    isn't just lost.
    """
    match = _DISAMBIGUATOR_RE.match(title)
    if match is None:
        return title, None
    return match.group("base"), match.group("topic")


def _prefix_topic(text: str | None, topic: str | None) -> str | None:
    """Prepend a disambiguating topic (from `_split_title`) to `text`, e.g.
    ``"(mythology) Paris ... is a figure from Greek mythology"``.
    """
    if topic is None:
        return text
    return f"({topic}) {text}" if text else f"({topic})"


class WikipediaEntitySource(EntitySource):
    """Queries live Wikipedia, via the MediaWiki API, as an
    [EntitySource][linkingtk.core.source.EntitySource].

    Entity ids are Wikipedia page titles, kept exactly as Wikipedia has
    them (including any parenthetical disambiguator, e.g.
    ``"Paris (mythology)"``) so `get` can round-trip them. Labels, however,
    have that disambiguator stripped (e.g. ``"Paris"``) with the topic
    moved to the front of the description instead (e.g. ``"(mythology) ..."``)
    -- see `_split_title` -- so that a bare-word mention can actually reach
    a disambiguated page as a candidate through
    [ExactMatch][linkingtk.blocking.exact.ExactMatch]'s exact-label
    blocking.

    Args:
        lang: The Wikipedia language edition to query (e.g. ``"en"`` for
            `en.wikipedia.org`).
        session: A `requests.Session` to issue requests through. Defaults to
            a fresh one carrying a descriptive `User-Agent` (MediaWiki API
            etiquette requires one, and `requests.Session`'s own default
            `User-Agent` gets rejected outright -- confirmed directly: the
            live API 403s a request left at `requests`' default
            ``"python-requests/*"`` header). A caller-supplied session is
            used exactly as given, headers untouched -- set your own
            `User-Agent` on it. Pass a shared session to reuse connections,
            or a test double to avoid real network calls.

    Raises:
        OptionalDependencyError: If `requests` isn't installed.
    """

    def __init__(self, lang: str = "en", session: requests.Session | None = None) -> None:
        try:
            import requests
        except ImportError as exc:
            raise OptionalDependencyError("WikipediaEntitySource", "wikipedia") from exc
        self.lang = lang
        self._api_url = f"https://{lang}.wikipedia.org/w/api.php"
        if session is not None:
            self._session = session
        else:
            self._session = requests.Session()
            self._session.headers["User-Agent"] = _USER_AGENT

    def search(self, query: str, top_k: int = 10) -> list[Entity]:
        """Full-text search Wikipedia for `query`, best match first.

        Uses `action=query&list=search` -- each hit's ``snippet`` (an
        HTML-highlighted excerpt around the matched terms) becomes the
        returned entity's description, with the highlighting markup
        stripped.
        """
        params: dict[str, str | int] = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": top_k,
            "format": "json",
        }
        response = self._session.get(self._api_url, params=params, timeout=10)
        response.raise_for_status()
        hits = response.json()["query"]["search"]
        entities = []
        for hit in hits:
            label, topic = _split_title(hit["title"])
            entities.append(
                Entity(
                    id=hit["title"],
                    labels=[label],
                    description=_prefix_topic(_clean_snippet(hit["snippet"]), topic),
                )
            )
        return entities

    def get(self, entity_id: str) -> Entity | None:
        """Look up a Wikipedia page by title, or ``None`` if it doesn't exist.

        Uses `action=query&prop=extracts` to fetch the page's intro
        paragraph as plain text.
        """
        params: dict[str, str | int] = {
            "action": "query",
            "prop": "extracts|pageprops",
            "titles": entity_id,
            "exintro": 1,
            "explaintext": 1,
            "format": "json",
        }
        response = self._session.get(self._api_url, params=params, timeout=10)
        response.raise_for_status()
        pages = response.json()["query"]["pages"]
        page = next(iter(pages.values()))
        if "missing" in page:
            return None
        label, topic = _split_title(page["title"])
        return Entity(
            id=page["title"],
            labels=[label],
            description=_prefix_topic(page.get("extract") or None, topic),
        )
