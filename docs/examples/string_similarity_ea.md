# Entity Alignment with StringSimilarityLinker

This aligns two small knowledge graphs from [`ToyEADataset`](../datasets.md#toy-datasets)
describing the same three cities, but `kg2`'s labels append the country name
(`"Paris"` vs. `"Paris, France"`), so an exact label match would miss every
pair. [`LabelOverlap`](../reference/blocking.md) blocking (character n-gram
overlap) finds the candidates despite the mismatch, and
[`StringSimilarityLinker`](../reference/algorithms.md) picks the best one per
source entity by Jaccard token overlap on the `label` field.
[`LeskLinker`](../reference/algorithms.md) (see [the Lesk example](lesk_wsd.md))
is a preconfigured instance of this same class (`source_field="context"`,
`target_field="description"`, `metric="word_overlap"`).

```python
--8<-- "examples/string_similarity_ea.py"
```

Run with:

```bash
uv run python examples/string_similarity_ea.py
```

```text
kg1:paris -> kg2:paris (score=0.50)
kg1:berlin -> kg2:berlin (score=0.50)
kg1:rome -> kg2:rome (score=0.50)
Metrics: {'precision@1': 1.0, 'recall': 1.0, 'f1': 1.0}
```
