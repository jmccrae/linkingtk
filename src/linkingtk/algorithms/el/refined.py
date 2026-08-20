"""ReFinED-style bi-encoder Entity Linking.

ReFinED (Ayoola et al., "ReFinED: An Efficient Zero-shot-capable Approach
to End-to-End Entity Linking", 2022,
https://github.com/amazon-science/ReFinED) is impractical to wrap
directly: it isn't on PyPI (GitHub-zip install only), is pinned to Python
3.8 (this repo requires >=3.10), and needs GB-scale Wikipedia model
downloads on first use. This module instead builds a custom bi-encoder
that follows ReFinED's real distinguishing architectural idea -- a single
shared/tied transformer encoder that scores both the mention-in-context
and the candidate-entity representation through the same weights (unlike
BLINK's two-stage separate-encoder-plus-cross-encoder-reranker design) --
trained via [Trainer][linkingtk.train.trainer.Trainer], not the original
package.

See [ReFinEDEncoder][linkingtk.algorithms.el.refined.ReFinEDEncoder] and
[ReFinEDLinker][linkingtk.algorithms.el.refined.ReFinEDLinker].
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as functional
from torch import nn

from linkingtk.algorithms.base import DEFAULT_BLOCKING, BaseLinker
from linkingtk.algorithms.matching import DEFAULT_MATCHER, Matcher
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity, description_text, label_texts
from linkingtk.core.result import AlignmentResult
from linkingtk.core.source import EntitySource
from linkingtk.utils.graph import Graph

_MENTION_START = "[E]"
_MENTION_END = "[/E]"
# Chars kept each side of the mention span before marking it -- bounds the
# marked text to a size that reliably survives tokenizer truncation
# regardless of the source document's length (see _mention_text).
_CONTEXT_WINDOW_CHARS = 100


def _entity_text(entity: Entity) -> str:
    """``label + description`` for a KB entry -- same joined-field convention as
    [feature_classifier._combined_text][linkingtk.algorithms.feature_classifier]."""
    return " ".join(label_texts(entity)) + " " + description_text(entity)


def _mention_text(entity: Entity) -> str:
    """Mark the mention span with ``[E] ... [/E]`` markers, in a local window.

    Uses the ``(text, start, end)`` offsets when ``context`` is a
    [ContextWithSpan][linkingtk.core.entity.ContextWithSpan], but only the
    ``_CONTEXT_WINDOW_CHARS`` characters immediately surrounding the
    span -- not the full ``text``. ``text`` is often an entire source
    document (e.g. [AidaConllDataset][linkingtk.datasets.aida_conll.AidaConllDataset]'s
    full news articles, averaging ~250 tokens with mentions at a median
    offset of ~125 tokens into it); tokenizing the whole thing under
    ``encode()``'s bounded ``max_length`` would truncate away the ``[E]``/``[/E]``
    markers themselves for most mentions -- measured at 68% of
    AIDA-CoNLL's test-split mentions before this windowing was added (see
    issue #45's benchmark investigation). Falls back to wrapping the label
    and prepending it to the plain context string when ``context`` carries
    no offsets (e.g. [ToyELDataset][linkingtk.datasets.toy.ToyELDataset]'s
    fixture, where ``text`` is already mention-scale).
    """
    context = entity.context
    if isinstance(context, tuple):
        text, start, end = context
        window_start = max(0, start - _CONTEXT_WINDOW_CHARS)
        window_end = min(len(text), end + _CONTEXT_WINDOW_CHARS)
        local_start, local_end = start - window_start, end - window_start
        window = text[window_start:window_end]
        return (
            f"{window[:local_start]}{_MENTION_START} {window[local_start:local_end]} "
            f"{_MENTION_END}{window[local_end:]}"
        )
    label = " ".join(label_texts(entity))
    context_str = context if isinstance(context, str) else ""
    return f"{_MENTION_START} {label} {_MENTION_END} {context_str}".strip()


def _format_entity(entity: Entity) -> str:
    return _mention_text(entity) if entity.context is not None else _entity_text(entity)


def _mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


class ReFinEDEncoder(nn.Module):
    """Shared/tied transformer bi-encoder for Entity Linking.

    Satisfies [TrainableModel][linkingtk.train.trainer.TrainableModel]:
    `encode()` runs *both* mentions and KB entries through the same
    transformer weights, distinguishing the two purely by how each
    entity's text is formatted (mention-span markers vs. label+description)
    -- ReFinED's real distinguishing idea versus BLINK's two independently
    encoded sides.

    Args:
        model_name: A Hugging Face model id, loaded via
            ``AutoModel``/``AutoTokenizer.from_pretrained``. Defaults to
            ``distilbert-base-uncased``, matching
            [MultiKELinker][linkingtk.algorithms.ea.multike.MultiKELinker]'s
            transformer-default precedent. Pass a tiny test model (e.g.
            ``hf-internal-testing/tiny-random-DistilBertModel``, as
            ``test_multike.py`` already does) for fast, network-light tests.
        embedding_dim: Output embedding dimension, independent of the
            backbone's native hidden size.
        max_length: Maximum token length per encoded text (truncated).
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        embedding_dim: int = 256,
        max_length: int = 96,
    ) -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [_MENTION_START, _MENTION_END]}
        )
        self.transformer = AutoModel.from_pretrained(model_name)
        self.transformer.resize_token_embeddings(len(self.tokenizer))
        self.proj = nn.Linear(self.transformer.config.hidden_size, embedding_dim)
        self.max_length = max_length

    def encode(self, entities: list[Entity]) -> torch.Tensor:
        """Return a ``(len(entities), embedding_dim)`` L2-normalized embedding tensor."""
        texts = [_format_entity(entity) for entity in entities]
        device = self.proj.weight.device
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)
        hidden_states = self.transformer(**encoded).last_hidden_state
        pooled = _mean_pool(hidden_states, encoded["attention_mask"])
        return functional.normalize(self.proj(pooled), dim=-1)


class ReFinEDLinker(BaseLinker):
    """Bi-encoder Entity Linking scored by a shared/tied
    [ReFinEDEncoder][linkingtk.algorithms.el.refined.ReFinEDEncoder].

    Unlike every other linker in this repo, this has no ``fit()`` --
    ``self.encoder`` is a plain
    [TrainableModel][linkingtk.train.trainer.TrainableModel]; train it via
    [Trainer][linkingtk.train.trainer.Trainer] directly (matching
    DESIGN.md's own documented ``Trainer(model=linker_model, ...)``
    interface), then call ``link()``:

    ```python
    linker = ReFinEDLinker()
    trainer = Trainer(model=linker.encoder, args=args, train_data=pairs, eval_data=eval_pairs)
    trainer.train()
    results = linker.link(mentions, kb_entries)
    ```

    ``link()`` runs (just poorly) on an untrained encoder too -- there's
    no ``_fitted`` flag, since nothing here can observe whether
    ``Trainer.train()`` was ever called on ``self.encoder``.

    Args:
        model_name: Forwarded to
            [ReFinEDEncoder][linkingtk.algorithms.el.refined.ReFinEDEncoder].
        embedding_dim: Forwarded to ``ReFinEDEncoder``.
        max_length: Forwarded to ``ReFinEDEncoder``.
        matching: Strategy used to resolve scored candidates into final
            links. Defaults to
            [GreedyMatcher][linkingtk.algorithms.matching.GreedyMatcher].
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        embedding_dim: int = 256,
        max_length: int = 96,
        matching: Matcher = DEFAULT_MATCHER,
    ) -> None:
        self.encoder = ReFinEDEncoder(model_name, embedding_dim, max_length)
        self.matching = matching

    def link(
        self,
        dataset1: list[Entity],
        dataset2: list[Entity] | EntitySource,
        graph: Graph = None,
        blocking: BlockingStrategy = DEFAULT_BLOCKING,
    ) -> list[AlignmentResult]:
        pairs = blocking.candidate_pairs(dataset1, dataset2)
        if not pairs:
            return []

        mentions_by_id = {entity1.id: entity1 for entity1, _ in pairs}
        entities_by_id = {entity2.id: entity2 for _, entity2 in pairs}
        mention_ids = list(mentions_by_id)
        entity_ids = list(entities_by_id)

        with torch.no_grad():
            self.encoder.eval()
            mention_emb = self.encoder.encode([mentions_by_id[i] for i in mention_ids])
            entity_emb = self.encoder.encode([entities_by_id[i] for i in entity_ids])
            self.encoder.train()

        mention_row = {entity_id: row for row, entity_id in enumerate(mention_ids)}
        entity_row = {entity_id: row for row, entity_id in enumerate(entity_ids)}
        similarities = mention_emb @ entity_emb.T

        candidates_by_source: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for entity1, entity2 in pairs:
            score = similarities[mention_row[entity1.id], entity_row[entity2.id]]
            candidates_by_source[entity1.id].append((entity2.id, float(score)))

        return self.matching.match(candidates_by_source)
