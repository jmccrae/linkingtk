# Entity Alignment with a knowledge-graph-embedding linker

This aligns two small, isomorphic knowledge graphs -- two 4-entity chains
linked by a `next` relation -- by training a TransE embedding
([`KGELinker`](../reference/algorithms.md)) jointly over both graphs plus
the known alignment pairs added as extra "seed" triples, then scoring
unaligned candidates by cosine similarity of their trained embeddings.
Unlike [the feature-classifier example](feature_classifier_ea.md),
matching here comes entirely from graph *structure*, not entity
labels/text -- pykeen has no built-in Entity Alignment mode, so this is
the standard trick (MTransE/IPTransE-style): fold the known alignment
pairs into the graph as extra triples so joint training pulls aligned
entities' embeddings together, then read off unaligned pairs by embedding
similarity.

This is a pipeline-correctness demo, not a generalization benchmark:
every ground-truth pair is given to `fit()` as a seed, so it demonstrates
the KGE training/scoring pipeline works correctly, not how well it
aligns entities it wasn't directly told about.

Requires the `kge` optional dependency group (for `pykeen`) — install
with `uv sync --extra kge`.

```python
--8<-- "examples/kge_ea.py"
```

Run with:

```bash
uv run python examples/kge_ea.py
```

```text
kg1:a -> kg2:w (score=0.36)
kg1:b -> kg2:x (score=0.30)
kg1:c -> kg2:y (score=0.28)
kg1:d -> kg2:z (score=0.52)
Metrics: {'precision@1': 1.0, 'recall': 1.0, 'f1': 1.0}
```
