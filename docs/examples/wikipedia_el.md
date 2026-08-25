# Entity Linking against live Wikipedia

Disambiguates the mention "Paris" between the city and the Greek mythology
figure, without downloading or materializing any Wikipedia dump:
[`WikipediaEntitySource`](../reference/sources.md) queries the public
MediaWiki search API on demand. Wikipedia disambiguates same-named articles
via a parenthetical title suffix (e.g. `"Paris (mythology)"`) rather than
distinct labels the way WordNet lemmas do, so `WikipediaEntitySource` strips
that suffix out of each result's label and into its description instead --
otherwise [`ExactMatch`](../reference/blocking.md)'s exact-label blocking
would never treat a disambiguated page as a candidate for the bare mention
text at all. From there,
[`StringSimilarityLinker`](../reference/algorithms.md) scores context
against each candidate's snippet -- the same Lesk-style word-overlap scoring
as [WSD against live WordNet](wn_wsd.md), applied to EL instead.

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
m1 -> Paris (mythology) (score=3.0)
  alternatives: ['Paris', 'Paris (disambiguation)']
  intro: (mythology) Paris (Ancient Greek: Πάρις, romanized: Páris), also
  known as Alexander (Ancient Greek: Ἀλέξανδρος, romanized: Aléxandros), is
  a figure from Greek mythology who appears in the numerous stories about
  the Trojan War, including the Iliad. He was prince of Troy, son of King
  Priam and Queen Hecuba, and younger brother of Prince Hector. His
  elopement with Helen sparks the Trojan War, during which he fatally
  wounds Achilles.
```
