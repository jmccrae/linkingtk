"""EWISER Word Sense Disambiguation.

EWISER (Bevilacqua & Navigli, "Breaking Through the 80% Glass Ceiling:
Raising the State of the Art in Word Sense Disambiguation by Incorporating
Knowledge Graph Information", ACL 2020,
https://github.com/SapienzaNLP/ewiser) is a full-inventory sense
classifier, architecturally very different from
[GlossBertLinker][linkingtk.algorithms.wsd.glossbert.GlossBertLinker]'s
pairwise cross-encoder: it encodes a whole sentence **once** with a
**frozen** BERT, projects every word's hidden state through a small FFN to
raw logits over the *entire* WordNet sense inventory, then applies one
linear WordNet-relation graph-propagation step to those logits (see
[StructuredLogits][linkingtk.algorithms.wsd._ewiser_structured_logits.StructuredLogits])
before a mention's candidate senses are read off as indices into that one
shared output vector -- there's no gloss text involved in the forward pass
at all. `EwiserEncoder.score(pairs)` still returns the same
``(len(pairs),) -> torch.Tensor`` shape
[GlossBertEncoder.score][linkingtk.algorithms.wsd.glossbert.GlossBertEncoder.score]
does (so [EwiserLinker.link][linkingtk.algorithms.wsd.ewiser.EwiserLinker.link]
can reuse the same blocking -> score -> group -> match pattern every
linker in this package uses), but computes it very differently internally
-- see `EwiserEncoder.score`'s own docstring.

Reimplemented from the paper and from the tensor shapes/keys observed
directly in the three checkpoints released alongside it (not ported from
EWISER's own source, which is CC-BY-NC-SA 4.0 licensed -- incompatible
with redistributing literal code under this package's MIT license).
Confirmed by reading the reference's own model/criterion code directly
(not guessed):

- The BERT encoder is **never fine-tuned** -- every released checkpoint's
  own state dict holds only 11 tensors, all in the small decoder (2-layer
  FFN + the graph module); the encoder's weights come straight from a
  frozen `bert-large-cased`, loaded fresh from Hugging Face, not from the
  checkpoint.
- Only the **final** BERT layer's hidden state matters, despite the
  checkpoints' own recorded config listing 4 candidate layers
  (`context_embeddings_layers=[-4,-3,-2,-1]`) -- confirmed
  `context_embeddings_use_all_hidden=False` makes the model use only
  `inner_states[-1]`, i.e. plain `last_hidden_state`.
- Subword-to-word pooling uses this package's tokenizer's own fast
  `word_ids()` alignment (`is_split_into_words=True`) rather than
  reimplementing EWISER's bespoke subword-merging loop by hand -- an
  equivalent mean-pool, expressed through infrastructure the tokenizer
  already provides.
- `EwiserEncoder.from_checkpoint` reads a checkpoint's own baked-in sparse
  WordNet adjacency directly from its state dict -- no graph construction
  needed for checkpoint-based inference (see
  [build_relation_adjacency][linkingtk.algorithms.wsd._ewiser_graph.build_relation_adjacency]
  for the from-scratch-training path that does need one). Unpickling a
  checkpoint file requires tolerating references to training-only classes
  from its original (pre-release) package name, ``qbert`` -- see
  [load_fairseq_checkpoint][linkingtk.algorithms.wsd._ewiser_decoder.load_fairseq_checkpoint].

See [EwiserEncoder][linkingtk.algorithms.wsd.ewiser.EwiserEncoder] and
[EwiserLinker][linkingtk.algorithms.wsd.ewiser.EwiserLinker].
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
from torch import nn

from linkingtk.algorithms.base import DEFAULT_BLOCKING, BaseLinker
from linkingtk.algorithms.matching import DEFAULT_MATCHER, Matcher
from linkingtk.algorithms.wsd._ewiser_decoder import EwiserDecoder, load_fairseq_checkpoint
from linkingtk.algorithms.wsd._ewiser_text import (
    mean_pool_subwords,
    mention_sentence_and_span,
    whitespace_tokenize_with_offsets,
    word_index_for_span,
)
from linkingtk.algorithms.wsd._ewiser_vocab import SenseVocabulary
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity
from linkingtk.core.result import AlignmentResult
from linkingtk.core.source import EntitySource
from linkingtk.exceptions import LinkingTKError
from linkingtk.utils.graph import Graph


class EwiserEncoder(nn.Module):
    """Frozen-BERT full-inventory sense classifier with a WordNet-graph output layer.

    Args:
        model_name_or_path: A Hugging Face encoder id, e.g.
            ``"bert-large-cased"`` (the published checkpoints' own
            encoder). Pass a tiny test model for fast, network-light
            tests.
        vocabulary: The
            [SenseVocabulary][linkingtk.algorithms.wsd._ewiser_vocab.SenseVocabulary]
            the output layer classifies against.
        decoder_hidden_dim: Width of the decoder's single hidden layer.
            Defaults to ``512``, matching the reference.
        adjacency: An optional sparse ``[len(vocabulary), len(vocabulary)]``
            WordNet-relation adjacency (from
            [build_relation_adjacency][linkingtk.algorithms.wsd._ewiser_graph.build_relation_adjacency],
            or a checkpoint's own baked-in one via `from_checkpoint`). Wires
            up the graph-propagation step (`StructuredLogits`) if given;
            the model degrades to a plain (non-graph-aware) classifier if
            omitted.
        structured_logits_trainable: Whether the graph's edge weights are
            updated during training.
        structured_logits_renormalize: Forwarded to `StructuredLogits`.
        freeze_encoder: Whether the BERT encoder's parameters are frozen.
            Defaults to `True`, matching the paper -- exposed as a real
            constructor flag rather than hardcoded since nothing else in
            this package forbids fine-tuning it, but the default preserves
            fidelity to EWISER's own recipe (see the module docstring).
        dropout: Decoder dropout rate. Checkpoint loading doesn't restore
            this (dropout has no learned parameters), so it only affects
            `EwiserTrainer`-driven training.
        max_length: Maximum subword sequence length per sentence
            (truncated).
        forward_batch_size: Number of distinct sentences `score()` batches
            through the encoder+decoder per forward pass -- unrelated to
            how many `(mention, candidate_sense)` pairs `score()` is
            called with (which may share far fewer distinct sentences, see
            `score`'s own docstring).

    Raises:
        LinkingTKError: If `vocabulary` isn't given.
    """

    def __init__(
        self,
        model_name_or_path: str = "bert-large-cased",
        vocabulary: SenseVocabulary | None = None,
        decoder_hidden_dim: int = 512,
        adjacency: torch.Tensor | None = None,
        structured_logits_trainable: bool = True,
        structured_logits_renormalize: bool = False,
        freeze_encoder: bool = True,
        dropout: float = 0.1,
        max_length: int = 512,
        forward_batch_size: int = 16,
    ) -> None:
        super().__init__()
        if vocabulary is None:
            raise LinkingTKError("EwiserEncoder requires a `vocabulary` (see SenseVocabulary).")
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        self.freeze_encoder = freeze_encoder
        for parameter in self.encoder.parameters():
            parameter.requires_grad = not freeze_encoder

        self.vocabulary = vocabulary
        self.max_length = max_length
        self.forward_batch_size = forward_batch_size
        self.decoder = EwiserDecoder(
            input_dim=self.encoder.config.hidden_size,
            hidden_dim=decoder_hidden_dim,
            vocab_size=len(vocabulary),
            adjacency=adjacency,
            structured_logits_trainable=structured_logits_trainable,
            structured_logits_renormalize=structured_logits_renormalize,
            dropout=dropout,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        offsets_path: str | Path,
        lexicon: str = "omw-en:1.4",
        model_name_or_path: str = "bert-large-cased",
        forward_batch_size: int = 16,
    ) -> EwiserEncoder:
        """Load one of the three EWISER checkpoints released alongside the paper.

        Args:
            checkpoint_path: Local path to a checkpoint file (e.g.
                ``ewiser.semcor.pt``) -- not bundled with this package.
            offsets_path: Local path to that checkpoint's own
                ``res/dictionaries/offsets.txt`` -- also not bundled (see
                [SenseVocabulary.from_offsets_file][linkingtk.algorithms.wsd._ewiser_vocab.SenseVocabulary.from_offsets_file]).
            lexicon: Forwarded to `SenseVocabulary.from_offsets_file`.
            model_name_or_path: The encoder to pair with the checkpoint's
                decoder weights. Defaults to ``"bert-large-cased"``,
                matching every released checkpoint's own recorded config.
            forward_batch_size: Forwarded to `EwiserEncoder.__init__`.

        Returns:
            An `EwiserEncoder` with the checkpoint's decoder weights (FFN
            + graph adjacency) loaded, paired with a freshly downloaded
            frozen encoder (never part of the checkpoint -- see the module
            docstring).

        Raises:
            LinkingTKError: If `offsets_path` doesn't resolve to the same
                vocabulary size as the checkpoint's own output layer --
                the surest sign it isn't the exact file this checkpoint
                was trained with.
        """
        data = load_fairseq_checkpoint(checkpoint_path)
        args = data["args"]
        state_dict = data["model"]

        vocabulary = SenseVocabulary.from_offsets_file(offsets_path, lexicon=lexicon)
        vocab_size = int(state_dict["decoder.logits.weight"].shape[0])
        if len(vocabulary) != vocab_size:
            raise LinkingTKError(
                f"offsets_path resolved to a {len(vocabulary)}-entry vocabulary, but "
                f"the checkpoint's own output layer has {vocab_size} entries -- "
                "offsets_path must be the exact res/dictionaries/offsets.txt this "
                "checkpoint was trained with."
            )

        adjacency = torch.sparse_coo_tensor(
            state_dict["decoder.structured_logits.adjacency_pars.0"],
            state_dict["decoder.structured_logits.adjacency_pars.1"],
            size=tuple(state_dict["decoder.structured_logits.adjacency_pars.2"].tolist()),
        )

        encoder = cls(
            model_name_or_path=model_name_or_path,
            vocabulary=vocabulary,
            decoder_hidden_dim=int(state_dict["decoder.linears.0.weight"].shape[0]),
            adjacency=adjacency,
            structured_logits_trainable=bool(
                getattr(args, "decoder_structured_logits_trainable", True)
            ),
            forward_batch_size=forward_batch_size,
        )
        encoder.decoder.load_state_dict(
            {key[len("decoder.") :]: value for key, value in state_dict.items()}
        )
        return encoder

    def score(self, pairs: list[tuple[Entity, Entity]]) -> torch.Tensor:
        """Return a ``(len(pairs),)`` tensor: each pair's raw sense-classification logit.

        Unlike a cross-encoder's `score`, this does **not** run one
        forward pass per pair. It groups `pairs` by each mention's unique
        sentence (see `mention_sentence_and_span`), runs the
        encoder+decoder once per distinct sentence (batched
        `forward_batch_size` sentences at a time), and reads each pair's
        score off as ``logits[mention_word_index, candidate_sense_index]``
        from that sentence's shared output. A candidate whose sense id
        isn't in `vocabulary`, or a mention whose span doesn't resolve to
        a word position, scores ``-inf`` (never spuriously wins a match).
        """
        if not pairs:
            return torch.empty(0)

        sentences: list[str] = []
        sentence_index: dict[str, int] = {}
        mention_word_index: dict[int, int | None] = {}
        for mention, _sense in pairs:
            text, start, end = mention_sentence_and_span(mention)
            if text not in sentence_index:
                sentence_index[text] = len(sentences)
                sentences.append(text)
            if id(mention) not in mention_word_index:
                tokens = whitespace_tokenize_with_offsets(text)
                mention_word_index[id(mention)] = word_index_for_span(tokens, start, end)

        all_logits = self._encode_sentences(sentences)

        scores: list[torch.Tensor] = []
        for mention, sense in pairs:
            text, _start, _end = mention_sentence_and_span(mention)
            logits = all_logits[sentence_index[text]]
            word_index = mention_word_index[id(mention)]
            sense_index = self.vocabulary.index_for(sense.id)
            if word_index is None or sense_index is None:
                scores.append(logits.new_tensor(float("-inf")))
            else:
                scores.append(logits[word_index, sense_index])
        return torch.stack(scores)

    def _encode_sentences(self, sentences: list[str]) -> list[torch.Tensor]:
        results: list[torch.Tensor] = []
        for start in range(0, len(sentences), self.forward_batch_size):
            results.extend(self._encode_chunk(sentences[start : start + self.forward_batch_size]))
        return results

    def _encode_chunk(self, sentences: list[str]) -> list[torch.Tensor]:
        device = next(self.parameters()).device
        words_per_sentence = [
            [word for word, _s, _e in whitespace_tokenize_with_offsets(text)] for text in sentences
        ]
        encoding = self.tokenizer(
            words_per_sentence,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)

        self.encoder.eval()  # EWISER's own encoder is always frozen+eval, even mid-training
        with torch.set_grad_enabled(not self.freeze_encoder):
            hidden = self.encoder(
                input_ids=encoding["input_ids"], attention_mask=encoding["attention_mask"]
            ).last_hidden_state

        num_words = [len(words) for words in words_per_sentence]
        word_ids_batch = [encoding.word_ids(batch_index=i) for i in range(len(sentences))]
        pooled = mean_pool_subwords(hidden, word_ids_batch, num_words)
        return [self.decoder(vectors.unsqueeze(0)).squeeze(0) for vectors in pooled]


class EwiserLinker(BaseLinker):
    """WSD linking scored by an
    [EwiserEncoder][linkingtk.algorithms.wsd.ewiser.EwiserEncoder].

    Unlike [GlossBertLinker][linkingtk.algorithms.wsd.glossbert.GlossBertLinker],
    this takes an already-constructed `EwiserEncoder` rather than a bare
    `model_name_or_path` string -- `EwiserEncoder` needs a `SenseVocabulary`
    (and usually an `adjacency`) at construction time, with no sane
    string-only default.

    ```python
    encoder = EwiserEncoder.from_checkpoint("ewiser.semcor.pt", "offsets.txt")
    linker = EwiserLinker(encoder)
    results = linker.link(mentions, senses)
    ```

    Or, for from-scratch/continued training, construct `EwiserEncoder`
    directly and train it via
    [EwiserTrainer][linkingtk.train.ewiser_trainer.EwiserTrainer].

    Args:
        model: The `EwiserEncoder` to score candidates with.
        matching: Strategy used to resolve scored candidates into final
            links. Defaults to
            [GreedyMatcher][linkingtk.algorithms.matching.GreedyMatcher].
    """

    def __init__(self, model: EwiserEncoder, matching: Matcher = DEFAULT_MATCHER) -> None:
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
