"""ESC Word Sense Disambiguation.

ESC (Barba, Pasini & Navigli, "ESC: Redesigning WSD with Extractive Sense
Comprehension", NAACL 2021, https://github.com/SapienzaNLP/esc) reframes
WSD as extractive span comprehension: a mention's context sentence (with
the target word bracketed by literal ``<classify>``/``</classify>``
markers, see [insert_classify_markers][linkingtk.algorithms.wsd._esc_text.insert_classify_markers])
is paired with *every* candidate sense's gloss, concatenated into one
sequence, and a standard extractive-QA head
(`transformers.AutoModelForQuestionAnswering`) predicts which candidate's
gloss span answers the "question" -- unlike
[GlossBertLinker][linkingtk.algorithms.wsd.glossbert.GlossBertLinker]'s
one-forward-pass-per-`(mention, candidate)` cross-encoder, or
[EwiserLinker][linkingtk.algorithms.wsd.ewiser.EwiserLinker]'s one-forward
-pass-per-sentence full-inventory classifier: ESC is one forward pass per
**mention**, jointly observing every one of that mention's candidates at
once.

Reimplemented from the paper and from understanding the reference's own
source (`esc/esc_pl_module.py`, `esc/utils/definitions_tokenizer.py`), not
ported from it -- the reference repo is CC-BY-NC-SA 4.0, incompatible with
redistributing literal code under this package's MIT license (same
constraint already documented in
[EwiserEncoder][linkingtk.algorithms.wsd.ewiser.EwiserEncoder]'s module
docstring for the EWISER port). Confirmed directly against the paper's own
published SemCor checkpoint (its Lightning `hyper_parameters`/`state_dict`,
not just the paper text):

- `transformer_model: facebook/bart-large`, `squad_head: False`,
  `use_special_tokens: False`, `use_pmask: False` -- the checkpoint uses
  the plain `AutoModelForQuestionAnswering` path, not any of the
  reference's custom XLNet/SQuAD-head/special-token variants. Its
  `state_dict` holds only `qa_model.model.*` (a stock BART encoder+decoder)
  and `qa_model.qa_outputs.{weight,bias}` (a `Linear(hidden_size, 2)`
  start/end head).
- `transformers`' `BartForQuestionAnswering.forward` runs the **full
  encoder-decoder** stack (not encoder-only) -- `decoder_input_ids`
  defaults to a right-shift of `input_ids` when omitted -- and applies the
  QA head to the **decoder's** last hidden state. A real architectural
  detail of this checkpoint, not an incidental implementation choice:
  simply calling `AutoModelForQuestionAnswering.from_pretrained(...)` with
  `input_ids`/`attention_mask` (and, for training,
  `start_positions`/`end_positions`) reproduces it exactly, with `.loss`
  computed internally as HF's own `(start_loss + end_loss) / 2`
  cross-entropy -- no hand-rolled loss needed (see
  [EscTrainer][linkingtk.train.esc_trainer.EscTrainer]).
- Candidate scoring uses the reference's own ``"probabilistic"``
  prediction type (the README's own example command, and what its
  headline numbers are reported under): `start_logprob[start_token] +
  end_logprob[end_token]` for each candidate's resolved gloss span, both
  taken from a `log_softmax` over the full (padded) sequence.

See [EscEncoder][linkingtk.algorithms.wsd.esc.EscEncoder] and
[EscLinker][linkingtk.algorithms.wsd.esc.EscLinker].
"""

from __future__ import annotations

import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch import nn

from linkingtk.algorithms.base import DEFAULT_BLOCKING, BaseLinker
from linkingtk.algorithms.matching import DEFAULT_MATCHER, Matcher
from linkingtk.algorithms.wsd._esc_text import (
    build_joint_sequence,
    candidate_gloss,
    insert_classify_markers,
    pad_encoded_sequences,
)
from linkingtk.algorithms.wsd._ewiser_text import mention_sentence_and_span
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity
from linkingtk.core.result import AlignmentResult
from linkingtk.core.source import EntitySource
from linkingtk.utils.graph import Graph


def _stub_pytorch_lightning() -> None:
    """Register a minimal `pytorch_lightning.utilities.parsing.AttributeDict`
    stand-in in `sys.modules` so `torch.load` can unpickle a Lightning
    checkpoint without the real (heavy, unrelated-to-inference)
    `pytorch_lightning` package installed -- the same "tolerate a
    training-only class from an unavailable package" move
    `load_fairseq_checkpoint` already makes for EWISER's `qbert`
    references. A no-op if `pytorch_lightning` is already importable.
    """
    try:
        import pytorch_lightning  # type: ignore[import-not-found]  # noqa: F401

        return
    except ImportError:
        pass

    if "pytorch_lightning.utilities.parsing" in sys.modules:
        return

    class AttributeDict(dict[str, Any]):
        def __getattr__(self, key: str) -> Any:
            try:
                return self[key]
            except KeyError as exc:
                raise AttributeError(key) from exc

        def __setattr__(self, key: str, value: Any) -> None:
            self[key] = value

    pl_module = types.ModuleType("pytorch_lightning")
    pl_utilities = types.ModuleType("pytorch_lightning.utilities")
    pl_parsing = types.ModuleType("pytorch_lightning.utilities.parsing")
    pl_parsing.AttributeDict = AttributeDict  # type: ignore[attr-defined]
    pl_utilities.parsing = pl_parsing  # type: ignore[attr-defined]
    pl_module.utilities = pl_utilities  # type: ignore[attr-defined]
    sys.modules["pytorch_lightning"] = pl_module
    sys.modules["pytorch_lightning.utilities"] = pl_utilities
    sys.modules["pytorch_lightning.utilities.parsing"] = pl_parsing


class EscEncoder(nn.Module):
    """Extractive-QA cross-encoder scorer for WSD.

    Unlike [GlossBertEncoder][linkingtk.algorithms.wsd.glossbert.GlossBertEncoder]
    (one forward pass per `(mention, candidate)` pair), `score()` runs one
    forward pass per **mention**, jointly observing every one of that
    mention's candidates present in `pairs` -- see `score`'s own docstring.

    Args:
        model_name_or_path: A Hugging Face question-answering model id
            (e.g. ``"facebook/bart-large"``, the paper's own backbone), or
            a local directory. Pass a tiny test model for fast,
            network-light tests.
        max_length: Maximum joint-sequence token length (truncated).
        forward_batch_size: Number of distinct mentions `score()` batches
            through the encoder per forward pass -- unrelated to how many
            `(mention, candidate_sense)` pairs `score()` is called with
            (which may share far fewer distinct mentions).
    """

    def __init__(
        self,
        model_name_or_path: str = "facebook/bart-large",
        max_length: int = 1024,
        forward_batch_size: int = 8,
    ) -> None:
        super().__init__()
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=True, add_prefix_space=True
        )
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(model_name_or_path)
        self.max_length = max_length
        self.forward_batch_size = forward_batch_size

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model_name_or_path: str = "facebook/bart-large",
        max_length: int = 1024,
        forward_batch_size: int = 8,
    ) -> EscEncoder:
        """Load the paper's own published Lightning checkpoint (e.g. the
        SemCor checkpoint linked from the reference README).

        Args:
            checkpoint_path: Local path to a `.ckpt` file.
            model_name_or_path: The backbone architecture to construct
                before loading the checkpoint's weights -- must match what
                the checkpoint was trained with (its own
                `hyper_parameters["transformer_model"]`; every published
                ESC checkpoint uses ``"facebook/bart-large"``).
            max_length: Forwarded to `EscEncoder.__init__`.
            forward_batch_size: Forwarded to `EscEncoder.__init__`.

        Returns:
            An `EscEncoder` with the checkpoint's QA-head weights loaded.
        """
        _stub_pytorch_lightning()
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        state_dict = {
            key[len("qa_model.") :]: value
            for key, value in checkpoint["state_dict"].items()
            if key.startswith("qa_model.")
        }
        encoder = cls(
            model_name_or_path=model_name_or_path,
            max_length=max_length,
            forward_batch_size=forward_batch_size,
        )
        encoder.qa_model.load_state_dict(state_dict)
        return encoder

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        start_positions: torch.Tensor | None = None,
        end_positions: torch.Tensor | None = None,
    ) -> Any:
        """Thin pass-through to the wrapped QA model -- used by
        [EscTrainer][linkingtk.train.esc_trainer.EscTrainer] for its
        `.loss` (HF's own `(start_loss + end_loss) / 2` cross-entropy) and
        by `score()` for its `.start_logits`/`.end_logits`.
        """
        return self.qa_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
        )

    def score(self, pairs: list[tuple[Entity, Entity]]) -> torch.Tensor:
        """Return a ``(len(pairs),)`` tensor: each pair's ``start_logprob +
        end_logprob`` for its candidate's resolved gloss span.

        Unlike a cross-encoder's `score`, this does **not** run one forward
        pass per pair: it groups `pairs` by mention, builds one joint
        sequence per mention (context sentence + every one of that
        mention's candidates' glosses, concatenated -- see
        `build_joint_sequence`), and runs the encoder once per mention
        (batched `forward_batch_size` mentions at a time). A candidate
        whose gloss span can't be resolved within the (possibly truncated)
        joint sequence scores ``-inf`` (never spuriously wins a match) --
        same convention `EwiserEncoder.score` uses for an unresolved
        candidate.
        """
        if not pairs:
            return torch.empty(0)

        mention_order: dict[int, int] = {}
        mentions: list[Entity] = []
        candidates_by_mention: dict[int, dict[str, Entity]] = {}
        pairs_by_mention: dict[int, list[int]] = {}
        for pair_index, (mention, sense) in enumerate(pairs):
            key = id(mention)
            if key not in mention_order:
                mention_order[key] = len(mentions)
                mentions.append(mention)
                candidates_by_mention[key] = {}
            candidates_by_mention[key].setdefault(sense.id, sense)
            pairs_by_mention.setdefault(mention_order[key], []).append(pair_index)

        scores: list[torch.Tensor | None] = [None] * len(pairs)
        for chunk_start in range(0, len(mentions), self.forward_batch_size):
            chunk_mentions = mentions[chunk_start : chunk_start + self.forward_batch_size]
            chunk_candidates = [
                list(candidates_by_mention[id(mention)].values()) for mention in chunk_mentions
            ]
            chunk_start_logprobs, chunk_end_logprobs, chunk_spans = self._encode_chunk(
                chunk_mentions, chunk_candidates
            )
            for offset in range(len(chunk_mentions)):
                mention_index = chunk_start + offset
                candidate_index_by_id = {
                    sense.id: index for index, sense in enumerate(chunk_candidates[offset])
                }
                for pair_index in pairs_by_mention.get(mention_index, []):
                    _mention, sense = pairs[pair_index]
                    candidate_index = candidate_index_by_id[sense.id]
                    span = chunk_spans[offset][candidate_index]
                    if span is None:
                        scores[pair_index] = chunk_start_logprobs[offset].new_tensor(float("-inf"))
                    else:
                        start_token, end_token = span
                        scores[pair_index] = (
                            chunk_start_logprobs[offset][start_token]
                            + chunk_end_logprobs[offset][end_token]
                        )

        assert all(score is not None for score in scores)  # every pair index was filled above
        return torch.stack(scores)  # type: ignore[arg-type]

    def _encode_chunk(
        self, mentions: list[Entity], candidates: list[list[Entity]]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[list[tuple[int, int] | None]]]:
        device = next(self.parameters()).device
        sequences = []
        spans_per_mention = []
        for mention, mention_candidates in zip(mentions, candidates, strict=True):
            text, start, end = mention_sentence_and_span(mention)
            marked = insert_classify_markers(text, start, end)
            glosses = [candidate_gloss(sense) for sense in mention_candidates]
            input_ids, attention_mask, spans = build_joint_sequence(
                self.tokenizer, marked, glosses, self.max_length
            )
            sequences.append((input_ids, attention_mask))
            spans_per_mention.append(spans)

        batch_input_ids, batch_attention_mask = pad_encoded_sequences(
            sequences, self.tokenizer.pad_token_id
        )
        batch_input_ids = batch_input_ids.to(device)
        batch_attention_mask = batch_attention_mask.to(device)

        outputs = self.forward(batch_input_ids, batch_attention_mask)
        start_logprobs = functional.log_softmax(outputs.start_logits, dim=-1)
        end_logprobs = functional.log_softmax(outputs.end_logits, dim=-1)
        return list(start_logprobs), list(end_logprobs), spans_per_mention


class EscLinker(BaseLinker):
    """WSD linking scored by an [EscEncoder][linkingtk.algorithms.wsd.esc.EscEncoder].

    ```python
    encoder = EscEncoder.from_checkpoint("escher_semcor_best.ckpt")
    linker = EscLinker(encoder)
    results = linker.link(mentions, senses)
    ```

    Or, for from-scratch/continued training, construct `EscEncoder`
    directly and train it via
    [EscTrainer][linkingtk.train.esc_trainer.EscTrainer].

    Args:
        model: The `EscEncoder` to score candidates with.
        matching: Strategy used to resolve scored candidates into final
            links. Defaults to
            [GreedyMatcher][linkingtk.algorithms.matching.GreedyMatcher].
    """

    def __init__(self, model: EscEncoder, matching: Matcher = DEFAULT_MATCHER) -> None:
        self.model = model
        self.matching = matching

    def link(
        self,
        dataset1: list[Entity],
        dataset2: list[Entity] | EntitySource,
        graph: Graph = None,
        blocking: BlockingStrategy = DEFAULT_BLOCKING,
    ) -> list[AlignmentResult]:
        pairs = list(blocking.candidate_pairs(dataset1, dataset2))
        if not pairs:
            return []

        with torch.no_grad():
            self.model.eval()
            scores = self.model.score(pairs)
            self.model.train()

        candidates_by_source: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for (entity1, entity2), score in zip(pairs, scores.tolist(), strict=True):
            candidates_by_source[entity1.id].append((entity2.id, score))

        return self.matching.match(candidates_by_source)
