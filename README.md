# LinkingTK

A Python toolkit that unifies four text-linking tasks behind a common
interface:

- **Entity Alignment (EA)** — matching entities between two knowledge graphs.
- **Entity Linking (EL)** — linking mentions in text to a knowledge base.
- **Word Sense Disambiguation (WSD)** — linking mentions in text to dictionary senses.
- **Word Sense Alignment (WSA)** — matching senses between two dictionaries.

See [`DESIGN.md`](./DESIGN.md) for the full design rationale, the `Entity`
and `BaseLinker` interfaces, dataset references, and the algorithm survey
for each task.

## Project layout

```
linkingtk/
├── data/                       # Local data files
├── src/linkingtk/
│   ├── core/                   # Entity, AlignmentResult
│   ├── blocking/                # BlockingStrategy, ExactMatch, ...
│   ├── algorithms/              # BaseLinker + ea/el/wsd/wsa submodules
│   ├── datasets/                # Dataset loaders & HF integrations
│   ├── utils/                   # NetworkX / RDFLib graph helpers
│   ├── train/                   # Trainer / TrainingArguments
│   └── eval/                    # Evaluator / EvaluationReport
├── tests/                      # Mirrors src/ layout
└── examples/                   # Runnable sample scripts
```

## Getting started

```bash
uv sync --extra graph --extra nlp --group dev
uv run pytest
uv run python examples/basic_exact_match.py
```

## Development

```bash
uv run ruff check .
uv run ruff format .
uv run mypy --strict src/linkingtk
```
