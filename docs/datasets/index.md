# Datasets

`linkingtk.datasets` provides ready-made [`DatasetLoader`](../reference/datasets.md)
implementations. Each one's `load()` returns a `(dataset1, dataset2,
ground_truth)` tuple, ready to hand straight to a linker's `link()` method
and then to [`Evaluator.evaluate`](../reference/eval.md). See the
navigation on the left for the individual dataset families.
