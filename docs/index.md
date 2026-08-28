# LinkingTK

LinkingTK is a Python toolkit that unifies four text-linking tasks behind a
common interface:

- **Entity Alignment (EA)** — matching entities between two knowledge graphs.
- **Entity Linking (EL)** — linking mentions in text to a knowledge base.
- **Word Sense Disambiguation (WSD)** — linking mentions in text to dictionary senses.
- **Word Sense Alignment (WSA)** — matching senses between two dictionaries.

See **[Getting Started](getting_started.md)** for installation and a
walkthrough of the core interface. This site is also the generated API
reference. For the design rationale, task-to-`Entity` mapping, dataset
references, and development roadmap, see
[DESIGN.md](https://github.com/jmccrae/linkingtk/blob/main/DESIGN.md).

## API Reference

- **[Core](reference/core.md)** — `Entity`, `AlignmentResult`.
- **[Blocking](reference/blocking.md)** — `BlockingStrategy` and its
  implementations (`ExactMatch`, `LabelOverlap`, `EmbeddingSimilarityBlocker`).
- **[Algorithms](reference/algorithms.md)** — `BaseLinker` and per-task
  linkers (e.g. `LeskLinker` for WSD).
- **[Datasets](reference/datasets.md)** — `DatasetLoader`.
- **[Utils](reference/utils.md)** — graph helpers wrapping NetworkX/RDFLib.
- **[Training](reference/train.md)** — `Trainer`, `TrainingArguments`.
- **[Evaluation](reference/eval.md)** — `Evaluator`, `EvaluationReport`.
- **[Exceptions](reference/exceptions.md)** — `LinkingTKError`.
