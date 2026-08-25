"""Entity Linking against live Wikipedia, via the MediaWiki API.

Links a mention straight to a Wikipedia page without downloading or
materializing any dump: [`WikipediaEntitySource`](../reference/sources.md)
queries the public MediaWiki search API on demand, and
[`ExactMatch`](../reference/blocking.md) narrows those results down to the
one whose title exactly matches the mention text before
[`StringSimilarityLinker`](../reference/algorithms.md) scores context
against each candidate's intro paragraph -- the same Lesk-style word-overlap
scoring used in `wn_wsd.py`, applied to EL instead of WSD.

Wrapped in [`CachingEntitySource`](../reference/core.md) since this hits a
real, rate-limited API.

Requires the `wikipedia` optional dependency and live network access:

    uv pip install linkingtk[wikipedia]

Run with: `uv run python examples/wikipedia_el.py`
"""

from linkingtk.algorithms.string_similarity import StringSimilarityLinker
from linkingtk.blocking import ExactMatch
from linkingtk.core import CachingEntitySource, Entity
from linkingtk.sources import WikipediaEntitySource


def main() -> None:
    mentions = [
        Entity(
            id="m1",
            labels=["Albert Einstein"],
            context="developed the theory of relativity and won the Nobel Prize in Physics",
        ),
    ]
    pages = CachingEntitySource(WikipediaEntitySource(lang="en"))

    linker = StringSimilarityLinker(
        source_field="context", target_field="description", metric="word_overlap"
    )
    results = linker.link(mentions, pages, blocking=ExactMatch())
    for result in results:
        print(f"{result.source_id} -> {result.target_id} (score={result.score})")

    target = pages.get(results[0].target_id)
    assert target is not None
    print(f"  intro: {target.description}")


if __name__ == "__main__":
    main()
