"""BLINK-style bi-encoder Entity Linking.

BLINK (Wu et al., "Scalable Zero-shot Entity Linking with Dense Entity
Retrieval", 2020, https://github.com/facebookresearch/BLINK) is
impractical to wrap directly: it isn't on PyPI (GitHub-only), and its
release is pinned to old ``transformers``/Python versions with GB-scale
Wikipedia model downloads. This module instead builds a custom bi-encoder
that follows BLINK's real distinguishing architectural idea -- **two
independently-parameterized transformers**, one for the mention-in-
context, one for the candidate entity's title+description, each scored
via ``[CLS]``-pooled dot product (Section 4.1 of the paper) -- unlike
[ReFinEDLinker][linkingtk.algorithms.el.refined.ReFinEDLinker]'s single
shared/tied encoder. Trained via
[Trainer][linkingtk.train.trainer.Trainer], not the original package.

Candidate generation (this module's ``link()``) only covers BLINK's
*first* stage; the paper's second-stage cross-encoder reranker is out of
scope here (see issue #23).

See [BlinkEncoder][linkingtk.algorithms.el.blink.BlinkEncoder] and
[BlinkLinker][linkingtk.algorithms.el.blink.BlinkLinker].
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as functional
from torch import nn

from linkingtk.algorithms.base import DEFAULT_BLOCKING, BaseLinker
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity, description_text, label_texts
from linkingtk.core.result import AlignmentResult
from linkingtk.core.source import EntitySource
from linkingtk.matchers import DEFAULT_MATCHER, Matcher
from linkingtk.utils.graph import Graph

_MENTION_START = "[Ms]"
_MENTION_END = "[Me]"
_ENTITY_SEP = "[ENT]"
# Chars kept each side of the mention span before marking it -- same fix as
# ReFinEDEncoder's _CONTEXT_WINDOW_CHARS (issue #45): without it, mean/CLS
# pooling a full source document under a bounded max_length truncates the
# span markers away entirely for mentions far into a long document.
_CONTEXT_WINDOW_CHARS = 100


def _entity_text(entity: Entity) -> str:
    """``title [ENT] description`` for a KB entry, per the paper's entity input format."""
    title = " ".join(label_texts(entity))
    return f"{title} {_ENTITY_SEP} {description_text(entity)}"


def _mention_text(entity: Entity) -> str:
    """Mark the mention span with ``[Ms] ... [Me]`` markers, in a local window.

    See [_mention_text][linkingtk.algorithms.el.refined._mention_text]'s
    docstring for why the window (not the full source text) is used.
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


class _SubEncoder(nn.Module):
    """One side of the bi-encoder: a transformer, tokenizer, and projection head."""

    def __init__(
        self, model_name: str, embedding_dim: int, extra_special_tokens: list[str]
    ) -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_special_tokens({"additional_special_tokens": extra_special_tokens})
        self.transformer = AutoModel.from_pretrained(model_name)
        self.transformer.resize_token_embeddings(len(self.tokenizer))
        self.proj = nn.Linear(self.transformer.config.hidden_size, embedding_dim)

    def encode_texts(self, texts: list[str], max_length: int, batch_size: int) -> torch.Tensor:
        device = self.proj.weight.device
        if not texts:
            return torch.empty(0, self.proj.out_features, device=device)

        chunks = []
        for start in range(0, len(texts), batch_size):
            encoded = self.tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            hidden_states = self.transformer(**encoded).last_hidden_state
            cls_pooled = hidden_states[:, 0, :]  # [CLS] token, per BLINK's red(.) choice
            chunks.append(functional.normalize(self.proj(cls_pooled), dim=-1))
        return torch.cat(chunks, dim=0)


class BlinkEncoder(nn.Module):
    """Two independently-parameterized transformer bi-encoder for Entity Linking.

    Satisfies [TrainableModel][linkingtk.train.trainer.TrainableModel]:
    `encode()` dispatches each entity to a mention-side or entity-side
    sub-transformer (by whether ``entity.context is not None``, mirroring
    [ReFinEDEncoder][linkingtk.algorithms.el.refined.ReFinEDEncoder]'s
    dispatch) -- unlike ``ReFinEDEncoder``, the two sides share no weights
    at all, BLINK's real distinguishing idea.

    Args:
        mention_model_name: Hugging Face model id for the mention-side
            encoder, loaded via ``AutoModel``/``AutoTokenizer.from_pretrained``.
            Defaults to ``distilbert-base-uncased``, matching
            [ReFinEDEncoder][linkingtk.algorithms.el.refined.ReFinEDEncoder]'s
            precedent.
        entity_model_name: Hugging Face model id for the entity-side
            encoder. Defaults to the same as ``mention_model_name``.
        embedding_dim: Output embedding dimension for both sides,
            independent of either backbone's native hidden size.
        max_length: Maximum token length per encoded text (truncated).
        encode_batch_size: Maximum number of texts forwarded through a
            sub-transformer in one go; ``encode()`` chunks larger inputs
            internally and concatenates the results, so its own signature
            is unaffected. Needed at real-dataset scale -- unlike
            [ReFinEDEncoder][linkingtk.algorithms.el.refined.ReFinEDEncoder],
            which has never been called with more than a few thousand
            entities at once, a single Zeshel domain's entity dictionary
            can have 100K+ entries, and forwarding all of them through a
            transformer in one unbatched call exhausts GPU memory.
    """

    def __init__(
        self,
        mention_model_name: str = "distilbert-base-uncased",
        entity_model_name: str | None = None,
        embedding_dim: int = 256,
        max_length: int = 96,
        encode_batch_size: int = 64,
    ) -> None:
        super().__init__()
        self.mention_encoder = _SubEncoder(
            mention_model_name, embedding_dim, [_MENTION_START, _MENTION_END]
        )
        self.entity_encoder = _SubEncoder(
            entity_model_name or mention_model_name, embedding_dim, [_ENTITY_SEP]
        )
        self.max_length = max_length
        self.encode_batch_size = encode_batch_size

    def encode(self, entities: list[Entity]) -> torch.Tensor:
        """Return a ``(len(entities), embedding_dim)`` L2-normalized embedding tensor.

        Mixes mention-side and entity-side entities in one call by
        encoding each group with its own sub-encoder and reassembling the
        result in the original order -- callers (e.g.
        [Trainer][linkingtk.train.trainer.Trainer]) never need to split a
        batch themselves.
        """
        mention_positions = [i for i, entity in enumerate(entities) if entity.context is not None]
        entity_positions = [i for i, entity in enumerate(entities) if entity.context is None]

        if not entity_positions:
            texts = [_mention_text(entities[i]) for i in mention_positions]
            return self.mention_encoder.encode_texts(texts, self.max_length, self.encode_batch_size)
        if not mention_positions:
            texts = [_entity_text(entities[i]) for i in entity_positions]
            return self.entity_encoder.encode_texts(texts, self.max_length, self.encode_batch_size)

        mention_emb = self.mention_encoder.encode_texts(
            [_mention_text(entities[i]) for i in mention_positions],
            self.max_length,
            self.encode_batch_size,
        )
        entity_emb = self.entity_encoder.encode_texts(
            [_entity_text(entities[i]) for i in entity_positions],
            self.max_length,
            self.encode_batch_size,
        )
        output = mention_emb.new_empty(len(entities), mention_emb.shape[-1])
        output[mention_positions] = mention_emb
        output[entity_positions] = entity_emb
        return output


class BlinkLinker(BaseLinker):
    """Bi-encoder Entity Linking scored by a two-tower
    [BlinkEncoder][linkingtk.algorithms.el.blink.BlinkEncoder].

    Like [ReFinEDLinker][linkingtk.algorithms.el.refined.ReFinEDLinker],
    this has no ``fit()`` -- ``self.encoder`` is a plain
    [TrainableModel][linkingtk.train.trainer.TrainableModel]; train it via
    [Trainer][linkingtk.train.trainer.Trainer] directly, then call
    ``link()``:

    ```python
    linker = BlinkLinker()
    trainer = Trainer(model=linker.encoder, args=args, train_data=pairs, eval_data=eval_pairs)
    trainer.train()
    results = linker.link(mentions, kb_entries)
    ```

    ``link()`` runs (just poorly) on an untrained encoder too -- there's
    no ``_fitted`` flag, since nothing here can observe whether
    ``Trainer.train()`` was ever called on ``self.encoder``.

    Args:
        mention_model_name: Forwarded to
            [BlinkEncoder][linkingtk.algorithms.el.blink.BlinkEncoder].
        entity_model_name: Forwarded to ``BlinkEncoder``.
        embedding_dim: Forwarded to ``BlinkEncoder``.
        max_length: Forwarded to ``BlinkEncoder``.
        matching: Strategy used to resolve scored candidates into final
            links. Defaults to
            [GreedyMatcher][linkingtk.matchers.greedy.GreedyMatcher].
    """

    def __init__(
        self,
        mention_model_name: str = "distilbert-base-uncased",
        entity_model_name: str | None = None,
        embedding_dim: int = 256,
        max_length: int = 96,
        matching: Matcher = DEFAULT_MATCHER,
    ) -> None:
        self.encoder = BlinkEncoder(
            mention_model_name, entity_model_name, embedding_dim, max_length
        )
        self.matching = matching

    def score_candidates(
        self,
        dataset1: list[Entity],
        dataset2: list[Entity] | EntitySource,
        blocking: BlockingStrategy = DEFAULT_BLOCKING,
    ) -> dict[str, list[tuple[str, float]]]:
        """Blocked candidates per source entity, scored but not yet matched.

        Satisfies [CandidateScorer][linkingtk.algorithms.llm_reranker.CandidateScorer]
        -- what `link()` itself
        builds internally, exposed so
        [LlmRerankerLinker][linkingtk.algorithms.llm_reranker.LlmRerankerLinker]
        (#23) can re-rank a narrowed top-k instead of every blocked pair.
        """
        pairs = blocking.candidate_pairs(dataset1, dataset2)
        if not pairs:
            return {}

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

        return candidates_by_source

    def link(
        self,
        dataset1: list[Entity],
        dataset2: list[Entity] | EntitySource,
        graph: Graph = None,
        blocking: BlockingStrategy = DEFAULT_BLOCKING,
    ) -> list[AlignmentResult]:
        return self.matching.match(self.score_candidates(dataset1, dataset2, blocking))
