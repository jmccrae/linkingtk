# Entity Linking against live Wikipedia

Links a mention straight to a Wikipedia page without downloading or
materializing any dump: [`WikipediaEntitySource`](../reference/sources.md)
queries the public MediaWiki search API on demand, and
[`ExactMatch`](../reference/blocking.md) narrows those results down to the
one whose title exactly matches the mention text before
[`StringSimilarityLinker`](../reference/algorithms.md) scores context
against each candidate's intro paragraph -- the same Lesk-style word-overlap
scoring as [WSD against live WordNet](wn_wsd.md), applied to EL instead.

Wrapped in [`CachingEntitySource`](../reference/core.md) since this hits a
real, rate-limited API.

Requires the `wikipedia` optional dependency and live network access:

```bash
uv pip install linkingtk[wikipedia]
```

```python
--8<-- "examples/wikipedia_el.py"
```

Run with:

```bash
uv run python examples/wikipedia_el.py
```

```text
m1 -> Albert Einstein (score=2.0)
  intro: Albert Einstein (14 March 1879 - 18 April 1955) was a German-born
  theoretical physicist best known for developing the theory of relativity.
  Einstein also made important contributions to quantum theory. His
  mass-energy equivalence formula E = mc2, which arises from special
  relativity, has been called "the world's most famous equation". He
  received the 1921 Nobel Prize in Physics for "his services to theoretical
  physics, and especially for his discovery of the law of the photoelectric
  effect".
  [...]
```

(The real intro extract runs to several more paragraphs -- truncated here
for readability.)
