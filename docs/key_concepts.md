# Key Concepts

LinkingTK models Entity Alignment, Entity Linking, Word Sense Disambiguation
and Word Sense Alignment as the same underlying problem: matching
[`Entity`][linkingtk.core.entity.Entity] objects from one dataset to
another. Four kinds of component cooperate to do that — a **Linker** drives
the whole process, calling on a **Blocker** to narrow the search space, a
**Source** to reach a dataset too large to load in full, and a **Matcher**
to turn scored candidates into final links.

## Linkers

A [`BaseLinker`][linkingtk.algorithms.base.BaseLinker] implements one task's
matching logic behind a single method:

```python
def link(
    self,
    dataset1: list[Entity],
    dataset2: list[Entity] | EntitySource,
    graph: Graph = None,
    blocking: BlockingStrategy = DEFAULT_BLOCKING,
) -> list[AlignmentResult]:
    ...
```

It takes two entity datasets, an optional supporting graph, and a
[Blocker](#blockers), and returns one
[`AlignmentResult`][linkingtk.core.result.AlignmentResult] per predicted
link — a `source_id` → `target_id` pair with a confidence `score`. Linkers
range from simple baselines
([`StringSimilarityLinker`][linkingtk.algorithms.string_similarity.StringSimilarityLinker])
to classical-ML classifiers
([`FeatureClassifierLinker`][linkingtk.algorithms.feature_classifier.FeatureClassifierLinker]),
knowledge-graph-embedding and deep-learning methods for EA
([`linkingtk.algorithms.ea`][]), transformer-based WSD
([`linkingtk.algorithms.wsd`][]), and LLM-backed linkers and re-rankers
([`LlmBaseLinker`][linkingtk.algorithms.llm.LlmBaseLinker],
[`LlmRerankerLinker`][linkingtk.algorithms.llm_reranker.LlmRerankerLinker]).
Internally, most linkers score candidate pairs and then hand the scores to
a [Matcher](#matchers) to resolve into final results.

## Blockers

A [`BlockingStrategy`][linkingtk.blocking.base.BlockingStrategy] generates
candidate pairs before any scoring happens, cutting the full
`dataset1 × dataset2` cross-product down to a smaller set worth comparing:

```python
def candidate_pairs(
    self, dataset1: list[Entity], dataset2: list[Entity] | EntitySource
) -> Iterable[tuple[Entity, Entity]]:
    ...
```

The default is [`ExactMatch`][linkingtk.blocking.exact.ExactMatch], which
pairs entities sharing an identical label.
[`LabelOverlap`][linkingtk.blocking.label_overlap.LabelOverlap] and
[`EmbeddingSimilarityBlocker`][linkingtk.blocking.embedding.EmbeddingSimilarityBlocker]
rank candidates by a graded similarity instead of requiring an exact
match, keeping only the top few per source entity.
[`Evaluator.evaluate_blocking`][linkingtk.eval.evaluator.Evaluator.evaluate_blocking]
scores a blocking pass on its own, independent of any downstream linker.

## Sources

An [`EntitySource`][linkingtk.core.source.EntitySource] stands in for a
`dataset2` too large — or already indexed elsewhere — to materialize as a
plain `list[Entity]`. Instead of enumerating the whole target set, a
linker or blocker queries it per mention:

```python
def search(self, query: str, top_k: int = 10) -> list[Entity]: ...
def get(self, entity_id: str) -> Entity | None: ...
```

Concrete sources wrap external targets such as WordNet
([`WnEntitySource`][linkingtk.sources.wn.WnEntitySource]), live Wikipedia
and Wikidata
([`WikipediaEntitySource`][linkingtk.sources.wikipedia.WikipediaEntitySource],
[`WikidataEntitySource`][linkingtk.sources.wikidata.WikidataEntitySource]),
and a local FAISS index
([`VectorIndexEntitySource`][linkingtk.sources.vector_index.VectorIndexEntitySource]).
Not every `BlockingStrategy` supports an `EntitySource` — one that needs to
enumerate the whole target set to build an index raises `TypeError`
instead; `ExactMatch` and `LabelOverlap` do support it, querying
`dataset2.search(...)` directly.
[`CachingEntitySource`][linkingtk.core.source.CachingEntitySource] wraps any
source to memoize `search`/`get` calls, worthwhile for HTTP-backed or
otherwise slow sources.

## Matchers

A [`Matcher`][linkingtk.matchers.base.Matcher] takes a linker's scored
candidates and resolves them into the final `AlignmentResult` list:

```python
def match(
    self, candidates_by_source: dict[str, list[tuple[str, float]]]
) -> list[AlignmentResult]:
    ...
```

[`GreedyMatcher`][linkingtk.matchers.greedy.GreedyMatcher] — the default —
picks each source entity's highest-scoring candidate independently, so two
sources may end up matched to the same target.
[`OptimalMatcher`][linkingtk.matchers.optimal.OptimalMatcher] instead finds
a single globally optimal one-to-one assignment via the Hungarian
algorithm, which can outperform greedy matching when two sources'
individually-best candidate is the same target. Linkers that produce a
scored candidate map accept a `matching` argument to choose between them —
e.g. `StringSimilarityLinker(matching=OptimalMatcher())`.

## Where to go next

- **[Getting Started](getting_started.md)** — install the package and run
  these four pieces together end-to-end.
- **[Examples](examples/index.md)** — task-specific linkers built on these
  same interfaces.
- **[API Reference](reference/core.md)** — full docs for every public class
  and function.
