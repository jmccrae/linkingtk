"""GlossBERT Word Sense Disambiguation.

GlossBERT (Huang et al., "GlossBERT: BERT for Word Sense Disambiguation
with Gloss Knowledge", EMNLP 2019,
https://github.com/HSLCY/GlossBERT) is a true cross-encoder: it
concatenates a mention's context sentence and a candidate sense's gloss
into one BERT input (`[CLS] context [SEP] gloss [SEP]`) and classifies
the pair as matching or not, unlike
[ReFinEDLinker][linkingtk.algorithms.el.refined.ReFinEDLinker]/
[BlinkLinker][linkingtk.algorithms.el.blink.BlinkLinker]'s bi-encoders
(independently `encode()` each side, then compare embeddings). Trained
via [CrossEncoderTrainer][linkingtk.train.cross_encoder.CrossEncoderTrainer],
built specifically for this shape.

This implements the paper's best-reported configuration,
**GlossBERT(Sent-CLS-WS)**, ported directly from the reference
implementation's own data-preparation code
(`preparation/generate_sent_cls_ws.py`), not just the paper text:

- Candidates are every WordNet sense of the mention's surface lemma,
  across *every* part of speech (not filtered to the mention's own gold
  POS) -- matches this repo's existing `ExactMatch`/`EntitySource.search()`
  design already.
- The gloss side of each pair is ``f"{mention_surface_text} : {gloss}"``
  -- the mention's own surface form, not the sense's dictionary lemma,
  confirmed against the reference's `wordnet/index.sense.gloss`.
- "Weak supervision" (the ``-WS`` in the paper's best model) wraps the
  mention span in the context sentence with literal ``"`` quote
  characters -- not special tokens. Confirmed by the paper's own
  published checkpoint (state dict `bert.*` + `classifier.weight (2, 768)`,
  vocab size 30522 -- stock ``bert-base-uncased``, nothing added).
- The classification head is `BertForSequenceClassification`-shaped
  (`num_labels=2`, `[CLS]`-pooled, softmax cross-entropy in the original)
  -- `score()` returns the class-1-minus-class-0 logit margin, which is
  mathematically identical to that 2-class softmax cross-entropy (so
  [CrossEncoderTrainer][linkingtk.train.cross_encoder.CrossEncoderTrainer]'s
  generic binary-cross-entropy-on-the-margin loss trains it exactly the
  same way, without a GlossBERT-specific loss function).

Because the architecture matches exactly, `GlossBertEncoder` also loads
the paper's own published checkpoint directly (pass its local directory
as `model_name_or_path`) -- see `examples/glossbert_reproduction.py` for
a real evaluation against it.

See [GlossBertEncoder][linkingtk.algorithms.wsd.glossbert.GlossBertEncoder]
and [GlossBertLinker][linkingtk.algorithms.wsd.glossbert.GlossBertLinker].
"""

from __future__ import annotations

from collections import defaultdict

import torch
from torch import nn

from linkingtk.algorithms.base import DEFAULT_BLOCKING, BaseLinker
from linkingtk.algorithms.matching import DEFAULT_MATCHER, Matcher
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity, description_text, label_texts
from linkingtk.core.result import AlignmentResult
from linkingtk.core.source import EntitySource
from linkingtk.utils.graph import Graph


def _mention_text(entity: Entity) -> str:
    """Wrap the mention span in literal ``"`` quote marks (GlossBERT's "weak supervision").

    Uses the ``(text, start, end)`` offsets when ``context`` is a
    [ContextWithSpan][linkingtk.core.entity.ContextWithSpan]. WSD mention
    contexts are single sentences (not whole documents like
    [AidaConllDataset][linkingtk.datasets.aida_conll.AidaConllDataset]'s),
    so unlike `refined.py`'s `_mention_text`, no truncation window is
    needed to keep the marked span inside `max_length`.
    """
    context = entity.context
    if isinstance(context, tuple):
        text, start, end = context
        return f'{text[:start]}" {text[start:end]} "{text[end:]}'
    label = " ".join(label_texts(entity))
    context_str = context if isinstance(context, str) else ""
    return f'" {label} " {context_str}'.strip()


def _mention_surface_text(entity: Entity) -> str:
    """The mention's literal surface text, e.g. ``"caught"`` -- as opposed to ``labels``,
    which (per [SemCorDataset][linkingtk.datasets.semcor.SemCorDataset]/
    [UfsacDataset][linkingtk.datasets.ufsac.UfsacDataset]) holds the
    dictionary lemma (e.g. ``"catch"``) instead, since that's what
    candidate generation needs to exact-match against WordNet.
    """
    context = entity.context
    if isinstance(context, tuple):
        text, start, end = context
        return text[start:end]
    return " ".join(label_texts(entity))


def _gloss_text(mention: Entity, sense: Entity) -> str:
    """``mention_surface_text : gloss`` -- the mention's own surface form, not the sense's label."""
    return f"{_mention_surface_text(mention)} : {description_text(sense)}"


class GlossBertEncoder(nn.Module):
    """`BertForSequenceClassification`-based cross-encoder scorer for WSD.

    Satisfies [CrossEncoderModel][linkingtk.train.cross_encoder.CrossEncoderModel].
    Unlike [ReFinEDEncoder][linkingtk.algorithms.el.refined.ReFinEDEncoder]/
    [BlinkEncoder][linkingtk.algorithms.el.blink.BlinkEncoder] (generic
    `AutoModel`, any backbone), this is hardcoded to BERT -- GlossBERT's
    own reference implementation is BERT-specific, and this is exactly
    what lets `model_name_or_path` point at the paper's own published
    checkpoint directory as well as a fresh ``bert-base-uncased``.

    Args:
        model_name_or_path: A Hugging Face model id, or a local directory
            (e.g. the paper's published `Sent_CLS_WS` checkpoint,
            extracted). Defaults to ``bert-base-uncased``, matching the
            paper. Pass a tiny test model (e.g.
            ``hf-internal-testing/tiny-random-BertForSequenceClassification``)
            for fast, network-light tests.
        max_length: Maximum token length for the tokenized ``(context,
            gloss)`` pair (truncated). The paper uses ``512``.
        forward_batch_size: Chunk size `score()` tokenizes/forwards at
            once, regardless of how many `pairs` it's called with --
            `GlossBertLinker.link()` and `CrossEncoderTrainer`'s eval step
            both pass every one of a mention's (or dataset's) candidate
            pairs to `score()` in a single call, which at
            `max_length=512` would exhaust GPU memory well before it
            reached even a few hundred real-corpus pairs if forwarded as
            one batch. Gradients still flow through every chunk normally
            (this only bounds how many pairs share one forward pass, not
            how many contribute to a loss) -- unrelated to
            `TrainingArguments.batch_size`, which bounds how many pairs
            *one gradient step* trains on.
    """

    def __init__(
        self,
        model_name_or_path: str = "bert-base-uncased",
        max_length: int = 512,
        forward_batch_size: int = 32,
    ) -> None:
        super().__init__()
        from transformers import BertForSequenceClassification, BertTokenizer

        self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
        self.model = BertForSequenceClassification.from_pretrained(model_name_or_path, num_labels=2)
        self.max_length = max_length
        self.forward_batch_size = forward_batch_size

    def score(self, pairs: list[tuple[Entity, Entity]]) -> torch.Tensor:
        """Return a ``(len(pairs),)`` tensor: the class-1-minus-class-0 logit margin per pair."""
        margins = [
            self._score_chunk(pairs[start : start + self.forward_batch_size])
            for start in range(0, len(pairs), self.forward_batch_size)
        ]
        return torch.cat(margins) if margins else torch.empty(0)

    def _score_chunk(self, pairs: list[tuple[Entity, Entity]]) -> torch.Tensor:
        context_texts = [_mention_text(mention) for mention, _sense in pairs]
        gloss_texts = [_gloss_text(mention, sense) for mention, sense in pairs]
        device = next(self.model.parameters()).device
        encoded = self.tokenizer(
            context_texts,
            gloss_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)
        logits = self.model(**encoded).logits
        margin: torch.Tensor = logits[:, 1] - logits[:, 0]
        return margin


class GlossBertLinker(BaseLinker):
    """WSD linking scored by a
    [GlossBertEncoder][linkingtk.algorithms.wsd.glossbert.GlossBertEncoder].

    Unlike every other linker in this repo, this has no ``fit()`` --
    ``self.model`` is a plain
    [CrossEncoderModel][linkingtk.train.cross_encoder.CrossEncoderModel];
    train it via [CrossEncoderTrainer][linkingtk.train.cross_encoder.CrossEncoderTrainer]
    directly, then call ``link()``:

    ```python
    linker = GlossBertLinker()
    trainer = CrossEncoderTrainer(model=linker.model, args=args, train_data=pairs)
    trainer.train()
    results = linker.link(mentions, senses)
    ```

    Or skip training entirely and load the paper's own published
    checkpoint: ``GlossBertLinker(model_name_or_path="/path/to/Sent_CLS_WS")``.

    Args:
        model_name_or_path: Forwarded to
            [GlossBertEncoder][linkingtk.algorithms.wsd.glossbert.GlossBertEncoder].
        max_length: Forwarded to ``GlossBertEncoder``.
        forward_batch_size: Forwarded to ``GlossBertEncoder``.
        matching: Strategy used to resolve scored candidates into final
            links. Defaults to
            [GreedyMatcher][linkingtk.algorithms.matching.GreedyMatcher].
    """

    def __init__(
        self,
        model_name_or_path: str = "bert-base-uncased",
        max_length: int = 512,
        forward_batch_size: int = 32,
        matching: Matcher = DEFAULT_MATCHER,
    ) -> None:
        self.model = GlossBertEncoder(model_name_or_path, max_length, forward_batch_size)
        self.matching = matching

    def score_candidates(
        self,
        dataset1: list[Entity],
        dataset2: list[Entity] | EntitySource,
        blocking: BlockingStrategy = DEFAULT_BLOCKING,
    ) -> dict[str, list[tuple[str, float]]]:
        """Blocked candidates per source entity, scored but not yet matched.

        Satisfies [CandidateScorer][linkingtk.algorithms.llm_reranker.CandidateScorer]
        -- what `link()` itself builds internally, exposed so
        [LlmRerankerLinker][linkingtk.algorithms.llm_reranker.LlmRerankerLinker]
        (#23) can re-rank a narrowed top-k instead of every blocked pair.
        """
        pairs = list(blocking.candidate_pairs(dataset1, dataset2))
        if not pairs:
            return {}

        with torch.no_grad():
            self.model.eval()
            scores = self.model.score(pairs)
            self.model.train()

        candidates_by_source: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for (entity1, entity2), score in zip(pairs, scores.tolist(), strict=True):
            candidates_by_source[entity1.id].append((entity2.id, score))

        return candidates_by_source

    def link(
        self,
        dataset1: list[Entity],
        dataset2: list[Entity] | EntitySource,
        graph: Graph = None,
        blocking: BlockingStrategy = DEFAULT_BLOCKING,
    ) -> list[AlignmentResult]:
        return self.matching.match(self.score_candidates(dataset1, dataset2, blocking))
