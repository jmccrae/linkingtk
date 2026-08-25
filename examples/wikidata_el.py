"""Entity Linking against live Wikidata, via its action API.

Disambiguates the mention "Paris" among several same-named Wikidata items --
the capital of France (Q90), a city in Texas (Q830149), a family name
(Q18331346), and a plant genus (Q162121) -- through
[`WikidataEntitySource`](../reference/sources.md)'s live `wbsearchentities`
search. Unlike Wikipedia, Wikidata doesn't fold disambiguation into the label
itself (every one of these items' label is plainly "Paris"), so
[`ExactMatch`](../reference/blocking.md) needs no special handling here --
see this example's sibling, `vector_index_el.py`, for the fully-offline
alternative for when live per-mention queries don't scale.

[`StringSimilarityLinker`](../reference/algorithms.md) then scores context
against each candidate's one-line Wikidata description to pick the right one.

Wrapped in [`CachingEntitySource`](../reference/core.md) since this hits a
real, rate-limited API.

Requires the `wikipedia` optional dependency (the HTTP client, shared with
[`WikipediaEntitySource`](../reference/sources.md)) and live network access:

    uv pip install linkingtk[wikipedia]

Run with: `uv run python examples/wikidata_el.py`
"""

from linkingtk.algorithms.string_similarity import StringSimilarityLinker
from linkingtk.blocking import ExactMatch
from linkingtk.core import CachingEntitySource, Entity
from linkingtk.sources import WikidataEntitySource


def main() -> None:
    mentions = [
        Entity(
            id="m1",
            labels=["Paris"],
            context="capital and most populous city of France, on the river Seine",
        ),
    ]
    items = CachingEntitySource(WikidataEntitySource(lang="en"))

    linker = StringSimilarityLinker(
        source_field="context", target_field="description", metric="word_overlap"
    )
    results = linker.link(mentions, items, blocking=ExactMatch())
    for result in results:
        print(f"{result.source_id} -> {result.target_id} (score={result.score})")
        print(f"  alternatives: {result.alternatives}")

    target = items.get(results[0].target_id)
    assert target is not None
    print(f"  description: {target.description}")


if __name__ == "__main__":
    main()
