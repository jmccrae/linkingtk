# LinkingTK

A Python toolkit that unifies four text-linking tasks behind a common
interface:

- **Entity Alignment (EA)** — matching entities between two knowledge graphs.
- **Entity Linking (EL)** — linking mentions in text to a knowledge base.
- **Word Sense Disambiguation (WSD)** — linking mentions in text to dictionary senses.
- **Word Sense Alignment (WSA)** — matching senses between two dictionaries.

See the generated documentation at
[john.mccr.ae/linkingtk](https://john.mccr.ae/linkingtk/) for a
Getting Started guide, the `Entity`/`BaseLinker` design (Key Concepts),
dataset references, and the full API reference.

## Project layout

```
linkingtk/
├── data/                       # Local data files
├── src/linkingtk/
│   ├── core/                   # Entity, AlignmentResult, EntitySource
│   ├── sources/                 # EntitySource wrappers (wn, ...)
│   ├── blocking/                # BlockingStrategy, ExactMatch, ...
│   ├── algorithms/              # BaseLinker + ea/el/wsd/wsa submodules
│   ├── datasets/                # Dataset loaders & HF integrations
│   ├── utils/                   # NetworkX / RDFLib graph helpers
│   ├── train/                   # Trainer / CrossEncoderTrainer / TrainingArguments
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

## Docs

```bash
uv sync --group docs
uv run mkdocs serve   # live preview at http://127.0.0.1:8000
uv run mkdocs build --strict
```

Docs are built and deployed to GitHub Pages automatically on every push to
`main` (see `.github/workflows/docs.yml`).
