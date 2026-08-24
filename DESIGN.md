# Linking Toolkit (LinkingTK)

## What is it and why do we need it?

LinkingTK is a Python-based package to support linking between different kinds
of text resources. It aims to unify four different tasks under a common interface

* **Entity Alignment (EA)**: This is the task of finding matches between two knowledge 
  graphs, where the output is generally a set of links between the identifiers.
  This is often a multilingual task, where the graphs are in different languages.
* **Entity Linking (EL)**: In this task, we aim to link occurrences of named entities
  in text to databases or knowledge graphs such as Wikipedia and Wikidata
* **Word Sense Disambiguation (WSD)**: Similarly, in this task, we attempt to identify
  the relevant sense in a dictionary. This is similar to EL, but the target is a
  dictionary
* **Word Sense Alignment (WSA)**: This task considers two dictionaries and 
  attempts to find equivalent senses between two entries. It can be monolingual
  or a harder multilingual variation, where the lemmas must also be mapped.

The core idea of LinkingTK is to define an entity as follows

```python
from dataclasses import dataclass, field
from typing import Union, Optional

@dataclass
class Entity:
    # An identifier for the entity
    id: str

    # Labels are either plain strings such as `"cat"`, or pairs with a language
    # such as `("cat", "en")`
    labels: list[Union[str, tuple[str, str]]]

    # The description is optional and corresponds to the main definition
    description: Optional[Union[str, tuple[str, str]]] = None

    # The context gives an occurence of the label in text. It may be given
    # with character offsets to indicate the location of the term
    context: Optional[Union[str, tuple[str, int, int]]] = None

    # Extra properties may be given to help describe the entity
    properties: dict[str, str] = field(default_factory=dict)
```

The main action would be the linking function that is prototyped below

```python
from abc import ABC, abstractmethod

class BaseLinker(ABC):
    @abstractmethod
    def link(dataset1 : list[Entity],
         dataset2 : list[Entity],
         graph : Union[list[tuple[str, str, str]], nx.Graph, rdflib.Graph, None] = None,
         blocking : BlockingStrategy = ExactMatch()) 
         -> list[AlignmentResult]:
        pass
```

This function takes two datasets, optionally a graph, supporting NetworkX, RDFLIB 
and plain triples, and a blocking strategy. The blocking strategy is used to find
initial matches, the default strategy, `ExactMatch`, assumes that only entities
with at least one matching label can be matches.

The tasks are mapped like this:

* **EA**: Entities have labels and descriptions, mostly not contexts. A graph is
          normally available. Blocking is mostly not exact, but allows some fuzzy
          matching.
* **EL**: Entities in `dataset1` have a context, but not a description; entities
          in `dataset2` have a description, but not a context. Blocking is mostly
          exact. A graph may be available but only covers entities in `dataset2`.
* **WSD**: Entities in `dataset1` have a context, but not a description; entities
          in `dataset2` have a description, often also a context. Blocking is always
          exact. A graph may be available but only covers entities in `dataset2`.
* **WSA**: Entities in datasets generally have descriptions and contexts. In the 
           monolingual case, blocking must be exact. Graphs are generally not 
           available.

### Blocking Strategies

```python
class BlockingStrategy(ABC):
    @abstractmethod
    def candidate_pairs(
        self, dataset1: list[Entity], dataset2: list[Entity]
    ) -> list[tuple[Entity, Entity]]:
        pass
```

List default strategies to implement first: `ExactMatch`, `LabelOverlap`, `EmbeddingSimilarityBlocker`.

## Project Architecture

```
linkingtk/
│
├── data/                       # Local data files
├── src/
│   └── linkingtk/
│       ├── __init__.py
│       ├── core/               # Entity, dataclasses, base interfaces
│       ├── blocking/           # ExactMatch, Fuzzy, etc.
│       ├── algorithms/         # EA, EL, WSD, WSA submodules
│       ├── datasets/           # Loaders & HuggingFace integrations
│       └── utils/              # Graph utilities (NetworkX, RDFLib wrappers)
│
├── tests/                      # Unit and integration tests mirroring src/
├── examples/                   # Runnable sample scripts
└── pyproject.toml              # Dependencies & packaging settings
```

## Core dependencies

Package management with `uv`

Core (required) dependencies cover the toolkit's baseline ML stack —
classical and neural methods used across multiple tasks. `[as a feature]`
dependencies are optional extras gating one specific algorithm family or
data format (graph-structured EA, spaCy-based NLP, KGE), installed only
when that family is used.

- networkx [as a feature]
- rdflib [as a feature]
- transformers
- datasets
- peft
- scikit-learn
- spacy [as a feature]
- pykeen [as a feature]

## Training & Fine-Tuning (`linkingtk.train`)

To support train/fine-tune workflows for trainable modules (e.g., GNNs for EA, Bi-Encoders for EL/WSD), `LinkingTK` includes a unified `Trainer` wrapper.

### Training Objectives & Loss Functions
- **Bi-Encoder / Contrastive Training**: Standard InfoNCE or Margin Ranking Loss taking source `Entity` and target `Entity` vectors.
- **Hard Negative Sampling**: Integrate `BlockingStrategy` to sample top-K non-matching entities as hard negatives during each batch generation.

### Interface
```python
from linkingtk.train import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="./models/my_el_model",
    learning_rate=3e-5,
    negative_samples_ratio=5,
    use_peft=True # Enable LoRA adapters for Transformer models
)

trainer = Trainer(
    model=linker_model,
    args=args,
    train_data=train_entities_pair,
    eval_data=val_entities_pair
)

trainer.train()
```

## Evaluation Module (`linkingtk.eval`)

`LinkingTK` provides a standardized evaluation interface tailored to the metrics traditionally used across Entity Alignment, Entity Linking, Word Sense Disambiguation, and Word Sense Alignment.

### Metric Definitions
- **Entity Alignment (EA)**: Defaults to Hits@1, Hits@10, and MRR.
- **Entity Linking (EL) & WSD**: Defaults to Micro Precision@1, Recall, and F1-score.
- **Word Sense Alignment (WSA)**: Defaults to Pairwise Precision, Recall, and F1-score.
- **Blocking Quality**: Calculates Pair Completeness (Blocking Recall) and Reduction Ratio.

### Interface
```python
from linkingtk.eval import Evaluator

# Simple evaluation
report = Evaluator.evaluate(
    predictions=[("e1", "e1_target"), ("e2", "e2_wrong")],
    ground_truth=[("e1", "e1_target"), ("e2", "e2_correct")]
)

print(report.metrics) 
# Output: {"precision@1": 0.5, "f1": 0.5, ...}

# Detailed ranking metrics (for EA algorithms)
report = Evaluator.evaluate_ranked(
    ranked_predictions=[("e1", ["e1_target", "e1_alt"]), ("e2", ["e2_wrong", "e2_correct"])],
    ground_truth=[("e1", "e1_target"), ("e2", "e2_correct")],
    top_k=[1, 5, 10]
)
print(report.metrics)
# Output: {"Hits@1": 0.5, "Hits@5": 1.0, "MRR": 0.75}

# Blocking-quality metrics (independent of any downstream linker)
report = Evaluator.evaluate_blocking(
    candidate_pairs=[("e1", "e1_target")],
    ground_truth=[("e1", "e1_target"), ("e2", "e2_correct")],
    dataset1_size=2,
    dataset2_size=2,
)
print(report.metrics)
# Output: {"pair_completeness": 0.5, "reduction_ratio": 0.75}
```

## Datasets

We should have loaders and supports for the main datasets across all four major
tasks. We should try to load datasets from their source using custom loaders,
but we may need to republish on HuggingFace Hub for some datasets

### Entity Alignment Datasets

#### Multilingual

- DBP15K
- https://github.com/nju-websoft/openea
  - EN-FR-15K, EN-DE-15K
  - EN-FR-100K, EN-DE-100K

##### Homogeneous Datasets

Homogeneous means linking between similar datasets, generally Wikipedia and 
derivatives such as Wikidata and YAGO

- https://github.com/nju-websoft/openea
  - D-W-15K, D-Y-15K
  - D-W-100K, D-Y-100K

##### Heterogeneous Datasets

Datasets from knowledge graphs created separately

- ICEWS (https://github.com/jxh4945777/Simple-HHEA/tree/main/data)
- WordNet-Wikidata (https://github.com/jmccrae/wn-wd-entity-align)

Toy datasets:

- https://github.com/insight-centre/naisc/tree/master/datasets/conference
- https://github.com/insight-centre/naisc/tree/master/datasets/anatomy

### Entity Linking Datasets

- AIDA-CoNLL: This has some questionable licensing... technically it is not 
   available but it is easy to find in fact
- TAC KBP: Genuinely LDC-licensed (LDC2018T16 + the LDC2014T16 reference
   KB), no free redistribution exists anywhere -- unlike AIDA-CoNLL/Zeshel,
   which route around licensing via a Hugging Face Hub community republish.
   Not implemented; would need a local LDC-obtained copy (`UfsacDataset`'s
   "local path required" precedent), not a fetchable default.
- ZESHEL: implemented, `linkingtk.datasets.zeshel.ZeshelDataset` (via the
   `naist-nlp/zeshel` Hugging Face Hub mirror)
- DaMuEL: https://lindat.mff.cuni.cz/repository/items/10e1cc03-b24b-4e41-9df9-b0fe4324ccbe
   -- implemented, `linkingtk.datasets.damuel.DamuelDataset`. Hosted on
   LINDAT/CLARIAH-CZ (DSpace 7 REST API), one `damuel_1.0_<lang>.tar` per
   language (278MB-26.3GB), each an uncompressed tar of 500 shuffled
   xz-compressed JSON-Lines shards -- streamed and sampled (`max_parts`),
   never fully downloaded.
- LCQuAD 2.0: https://figshare.com/projects/LCQuAD_2_0/62270

### Word Sense Disambiguation Datasets

- UFSAC: https://github.com/getalp/UFSAC
- SemCor 2026: https://github.com/globalwordnet/semcor/

### Word Sense Alignment

- https://sinaahmadi.github.io/resources/mwsa.html

## Algorithms

Generally, we would prefer a 'clean' integration rather than creating a 
spaghetti mess of dependencies. We should analyse this for each of these
before adding as a dependency of this project.

### Entity Alignment

- https://github.com/nju-websoft/openea
- https://github.com/jxh4945777/Simple-HHEA
- https://github.com/jxh4945777/ChatEA
- ProLEA: https://aclanthology.org/2025.findings-emnlp.1093.pdf
- https://github.com/DexterZeng/EntMatcher

More relevant papers/projects here: https://github.com/Xiefeng69/Awesome-Entity-Alignment

### Entity Linker

- https://spacy.io/api/entitylinker
- https://github.com/amazon-science/ReFinED
- https://github.com/facebookresearch/BLINK

More relevant projects here: https://github.com/topics/entity-linking?l=python&o=desc&s=stars

### Word Sense Disambiguation

- NLTK WSD (e.g., classic algorithms like Lesk)
- https://github.com/SapienzaNLP/ewiser
- https://github.com/HSLCY/GlossBERT
- https://github.com/SapienzaNLP/esc
- https://github.com/suytingwan/multilingual-WSD (maybe, needs checking)
- https://github.com/aalgirdas/wordnet_onto (maybe)

More relevant projects here: https://github.com/topics/word-sense-disambiguation?l=html&o=desc&s=forks

### Word Sense Alignment

- https://github.com/insight-centre/naisc

## Testing and Benchmarks

Use `pytest` to run tests. We should have at least one 'toy' dataset (<100 entities)
for each task to be used to test performance across all algorithms

## Milestones

- Phase 1: Core models (Entity, AlignmentResult), abstract base interfaces, and ExactMatch blocking.
- Phase 2: Evaluation metrics module & simple heuristic algorithms (e.g., exact label matching, Lesk for WSD).
- Phase 3: Dataset loader pipeline & Hugging Face integrations.
- Phase 4: Neural/Advanced algorithm wrappers (BLINK, ReFinED, OpenEA integration).

## Coding Standards & Documentation

To maintain clean, scalable, and idiomatic Python, all code generated for `LinkingTK` must adhere to the following standards:

### Code Quality & Tooling
- **Target Version:** Python 3.10+
- **Formatting & Linting:** Code must pass `ruff check .` and `ruff format .` without errors.
- **Type Checking:** All public APIs must pass `mypy --strict`. Use explicit type annotations (`list[Entity]`, `str | None`, etc.).
- **Modularity:** Avoid single files exceeding 300 lines. Split heavy submodules (e.g., individual algorithm wrappers) into dedicated modules.

Formatting and type checking, should be verified by GitHub CI script.

### Documentation Standard
- **Docstring Style:** Google-style docstrings for all public classes, methods, and functions.
- **Algorithm References:** Algorithmic modules must include paper links and citation references in the module docstring.
- **Package Extras:** Handle heavy/optional imports (e.g., `torch`, `rdflib`) gracefully:
  ```python
  try:
      import rdflib
  except ImportError:
      raise ImportError(
          "RDFLib is required for graph features. Install via 'pip install linkingtk[graph]'"
      )
```

### Logging & Error Handling

* Do not use `print()` statements in production code. Use `logging.getLogger("linkingtk")`.
* Define base package exceptions under `linkingtk.exceptions.LinkingTKError`.
