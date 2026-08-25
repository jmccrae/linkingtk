# Entity Linking against live Wikidata

Disambiguates the mention "Paris" among several same-named Wikidata items --
the capital of France (Q90), a city in Texas (Q830149), a family name
(Q18331346), and a plant genus (Q162121) -- through
[`WikidataEntitySource`](../reference/sources.md)'s live `wbsearchentities`
search. Unlike Wikipedia, Wikidata doesn't fold disambiguation into the label
itself (every one of these items' label is plainly "Paris"), so
[`ExactMatch`](../reference/blocking.md) needs no special handling here --
see this example's sibling, [a fully-offline vector index](vector_index_el.md),
for when live per-mention queries don't scale.

[`StringSimilarityLinker`](../reference/algorithms.md) then scores context
against each candidate's one-line Wikidata description to pick the right one.

Wrapped in [`CachingEntitySource`](../reference/core.md) since this hits a
real, rate-limited API.

Requires the `wikipedia` optional dependency (the HTTP client, shared with
[`WikipediaEntitySource`](../reference/sources.md)) and live network access:

```bash
uv pip install linkingtk[wikipedia]
```

```python
--8<-- "examples/wikidata_el.py"
```

Run with:

```bash
uv run python examples/wikidata_el.py
```

```text
m1 -> Q90 (score=5.0)
  alternatives: ['Q830149', 'Q1158980', 'Q162121', 'Q18331346', 'Q3181341']
  description: capital and most populous city in France
```
