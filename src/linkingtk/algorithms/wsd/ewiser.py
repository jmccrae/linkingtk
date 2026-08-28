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
does (so [EwiserLinker][linkingtk.algorithms.wsd.ewiser.EwiserLinker]'s ``link()``
can reuse the same blocking -> score -> group -> match pattern every
linker in this package uses), but computes it very differently internally
-- see `EwiserEncoder.score`'s own docstring.

Reimplemented from the paper and from the tensor shapes/keys observed
directly in the three checkpoints released alongside it (not ported from
EWISER's own source, which is CC-BY-NC-SA 4.0 licensed -- incompatible
with redistributing literal code under this package's Apache-2.0 license).
Confirmed by reading the reference's own model/criterion code directly
(not guessed):

- The BERT encoder is **never fine-tuned** -- every released checkpoint's
  own state dict holds only 11 tensors, all in the small decoder (2-layer
  FFN + the graph module); the encoder's weights come straight from a
  frozen `bert-large-cased`, loaded fresh from Hugging Face, not from the
  checkpoint.
- The encoder input is the **sum of the last 4 BERT hidden-state layers**
  (`context_embeddings_layers=[-4,-3,-2,-1]`), not just
  `last_hidden_state` -- despite every released checkpoint's own recorded
  `context_embeddings_use_all_hidden=False`, which would suggest
  otherwise. Confirmed this flag is dead: the reference's own
  `TaggerModel.build_model` computes `use_all_hidden` from
  `len(args.context_embeddings_type) > 1` (a variable-name mix-up --
  `context_embeddings_type` is the string `"bert"`, not the actual layer
  list), which for `"bert"` (length 4) is always `True`, unconditionally.
  See `EwiserEncoder`'s own `num_summed_layers` docstring for how this was
  traced.
- Each whitespace-split word is subword-tokenized **in isolation** via a
  from-scratch WordPiece implementation
  ([wordpiece_tokenize_words][linkingtk.algorithms.wsd._ewiser_text.wordpiece_tokenize_words]),
  not a single `tokenizer(words, is_split_into_words=True)` call --
  confirmed, by tracing a real checkpoint-reproduction gap layer by layer
  down to one divergent subtoken, that `is_split_into_words=True` still
  runs BERT's punctuation-splitting pre-tokenizer on each word first
  (splitting e.g. UFSAC's own single-token ``"Oct."`` into separate
  ``"Oct"``/``"."`` pre-tokens before wordpiece), producing a *standalone*
  ``"."`` token where EWISER's own reference (wordpiece-tokenizing the
  literal string ``"Oct."`` directly, no pre-splitting) produces a
  ``"##."`` continuation piece instead -- a different vocabulary entry
  with an unrelated embedding. See `wordpiece_tokenize_words`'s own
  docstring for the full diagnosis.
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
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from linkingtk.algorithms.base import DEFAULT_BLOCKING, BaseLinker
from linkingtk.algorithms.wsd._ewiser_decoder import EwiserDecoder, load_fairseq_checkpoint
from linkingtk.algorithms.wsd._ewiser_text import (
    mean_pool_subwords,
    mention_sentence_and_span,
    whitespace_tokenize_with_offsets,
    word_index_for_span,
    wordpiece_tokenize_words,
)
from linkingtk.algorithms.wsd._ewiser_vocab import SenseVocabulary
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity
from linkingtk.core.result import AlignmentResult
from linkingtk.core.source import EntitySource
from linkingtk.exceptions import LinkingTKError
from linkingtk.matchers import DEFAULT_MATCHER, Matcher
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
        output_embedding_init: An optional ``[len(vocabulary),
            decoder_hidden_dim]`` tensor to initialize
            ``decoder.logits.weight`` from, in place of its default random
            init -- see
            [load_synset_centroid_vectors][linkingtk.algorithms.wsd._ewiser_sense_embeddings.load_synset_centroid_vectors]/
            [build_synset_centroid_vectors_from_lmms][linkingtk.algorithms.wsd._ewiser_sense_embeddings.build_synset_centroid_vectors_from_lmms]
            to build one. This is what makes
            [EwiserTrainer.freeze_output_epochs][linkingtk.train.ewiser_trainer.EwiserTrainer]
            meaningful -- freezing a *pretrained* output layer for a few
            epochs protects it from noisy early gradients (the paper's own
            motivation); freezing the layer's plain default random init
            has no such benefit.
        num_summed_layers: The encoder input is the **sum** of the last
            `num_summed_layers` BERT hidden-state layers, not just the
            final layer -- confirmed against the reference's own
            `TaggerModel.build_model`, whose ``use_all_hidden=len(layers)
            > 1`` reads ``args.context_embeddings_type`` (a string, e.g.
            ``"bert"``) where it evidently meant to read
            ``args.context_embeddings_layers`` (the actual 4-entry layer
            list) -- since ``len("bert") == 4 > 1`` regardless of the
            real layer count, this makes ``use_all_hidden`` **always**
            true for every BERT-backed checkpoint, silently overriding
            the separate, correctly-named
            ``context_embeddings_use_all_hidden`` config flag (which
            reads `False` on all three released checkpoints and would,
            if honored, sum only the single last layer). A reference
            quirk, not a reference bug we get to ignore: it *is* what
            these checkpoints were actually trained against, confirmed by
            tracing a checkpoint-reproduction gap down to this exact
            vector (norm ~82 summed vs. ~18 single-layer, cosine 0.69
            apart -- unmistakably different vectors, not numerical noise).
            Defaults to ``4``, matching every released checkpoint's own
            ``context_embeddings_layers`` length.
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
        num_summed_layers: int = 4,
        max_length: int = 512,
        forward_batch_size: int = 16,
        output_embedding_init: torch.Tensor | None = None,
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
        self.num_summed_layers = num_summed_layers
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
        if output_embedding_init is not None:
            expected_shape = (len(vocabulary), self.decoder.logits.weight.shape[1])
            if tuple(output_embedding_init.shape) != expected_shape:
                raise LinkingTKError(
                    f"output_embedding_init has shape {tuple(output_embedding_init.shape)}, "
                    f"expected {expected_shape} (len(vocabulary), decoder_hidden_dim)."
                )
            self.decoder.logits.weight.data = output_embedding_init.to(
                self.decoder.logits.weight.dtype
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

        Each chunk's ``[num_words, len(vocabulary)]`` logits are consumed
        (scores extracted) and released before the next chunk is encoded,
        rather than holding every distinct sentence's full logits matrix
        in memory for the whole call -- at `len(vocabulary)` ~117k, a
        large evaluation set's worth of these held simultaneously
        (thousands of sentences at once, e.g. UFSAC's "ALL" split) is
        enough to exhaust even a 24GB GPU; confirmed by hitting exactly
        that `CUDA out of memory` running `examples/ewiser_reproduction.py`
        before this was fixed.
        """
        if not pairs:
            return torch.empty(0)

        sentences: list[str] = []
        sentence_index: dict[str, int] = {}
        mention_word_index: dict[int, int | None] = {}
        pairs_by_sentence: dict[int, list[int]] = {}
        for pair_index, (mention, _sense) in enumerate(pairs):
            text, start, end = mention_sentence_and_span(mention)
            if text not in sentence_index:
                sentence_index[text] = len(sentences)
                sentences.append(text)
            if id(mention) not in mention_word_index:
                tokens = whitespace_tokenize_with_offsets(text)
                mention_word_index[id(mention)] = word_index_for_span(tokens, start, end)
            pairs_by_sentence.setdefault(sentence_index[text], []).append(pair_index)

        scores: list[torch.Tensor | None] = [None] * len(pairs)
        for start in range(0, len(sentences), self.forward_batch_size):
            chunk_sentences = sentences[start : start + self.forward_batch_size]
            chunk_logits = self._encode_chunk(chunk_sentences)
            for offset, logits in enumerate(chunk_logits):
                for pair_index in pairs_by_sentence.get(start + offset, []):
                    mention, sense = pairs[pair_index]
                    word_index = mention_word_index[id(mention)]
                    sense_index = self.vocabulary.index_for(sense.id)
                    if word_index is None or sense_index is None:
                        scores[pair_index] = logits.new_tensor(float("-inf"))
                    else:
                        scores[pair_index] = logits[word_index, sense_index]

        assert all(score is not None for score in scores)  # every pair index was filled above
        return torch.stack(scores)  # type: ignore[arg-type]

    def _encode_chunk(self, sentences: list[str]) -> list[torch.Tensor]:
        device = next(self.parameters()).device
        words_per_sentence = [
            [word for word, _s, _e in whitespace_tokenize_with_offsets(text)] for text in sentences
        ]
        tokenized = [
            wordpiece_tokenize_words(self.tokenizer, words) for words in words_per_sentence
        ]
        # Truncate tokens/word_ids together so a truncated sentence's last
        # kept subtoken still has a matching word_ids entry.
        token_lists = [tokens[: self.max_length] for tokens, _word_ids in tokenized]
        word_ids_batch: list[list[int | None]] = [
            word_ids[: self.max_length] for _tokens, word_ids in tokenized
        ]
        max_len = max(len(tokens) for tokens in token_lists)
        pad_id = self.tokenizer.pad_token_id
        input_ids = torch.full((len(sentences), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(sentences), max_len), dtype=torch.long)
        for row, tokens in enumerate(token_lists):
            ids = [self.tokenizer.convert_tokens_to_ids(token) for token in tokens]
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, : len(ids)] = 1
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        self.encoder.eval()  # EWISER's own encoder is always frozen+eval, even mid-training
        with torch.set_grad_enabled(not self.freeze_encoder):
            hidden_states = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
            ).hidden_states
        hidden = torch.stack(hidden_states[-self.num_summed_layers :], dim=0).sum(0)

        num_words = [len(words) for words in words_per_sentence]
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
            [GreedyMatcher][linkingtk.matchers.greedy.GreedyMatcher].
    """

    def __init__(self, model: EwiserEncoder, matching: Matcher = DEFAULT_MATCHER) -> None:
        self.model = model
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

        If `dataset2` is a `WnEntitySource` whose `lexicon` differs from
        `model.vocabulary`'s own, candidate ids are translated into the
        vocabulary's lexicon via
        [synset_id_via_ili][linkingtk.sources.wn.synset_id_via_ili] before
        scoring (see `_translate_for_scoring`'s own docstring) -- without
        this, every candidate would fail
        [SenseVocabulary.index_for][linkingtk.algorithms.wsd._ewiser_vocab.SenseVocabulary.index_for]
        and silently score ``-inf`` (#67). Candidate ids in the returned
        dict are still `dataset2`'s own (untranslated) ids -- the
        translation is scoring-only, invisible to the caller.
        """
        pairs = list(blocking.candidate_pairs(dataset1, dataset2))
        if not pairs:
            return {}

        scoring_pairs = self._translate_for_scoring(pairs, dataset2)

        with torch.no_grad():
            self.model.eval()
            scores = self.model.score(scoring_pairs)
            self.model.train()

        candidates_by_source: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for (entity1, entity2), score in zip(pairs, scores.tolist(), strict=True):
            candidates_by_source[entity1.id].append((entity2.id, score))

        return candidates_by_source

    def _translate_for_scoring(
        self, pairs: list[tuple[Entity, Entity]], dataset2: list[Entity] | EntitySource
    ) -> list[tuple[Entity, Entity]]:
        """Replace each candidate's id with its `model.vocabulary`-lexicon counterpart,
        resolved via ILI, when `dataset2` is a `WnEntitySource` in a
        different lexicon than the vocabulary was built from -- a no-op
        (returns `pairs` unchanged) otherwise, including whenever
        `model.vocabulary.lexicon` is ``None`` (a `from_wn`-built
        vocabulary, not tied to any real `wn` lexicon at all).

        A candidate id that doesn't resolve via ILI (no ILI entry on
        either side, or no synset sharing it in the vocabulary's lexicon)
        is passed through untranslated, exactly like today -- `score()`'s
        own ``-inf`` fallback still applies, correctly, since that
        candidate genuinely isn't resolvable against this vocabulary.
        """
        from linkingtk.sources.wn import WnEntitySource, synset_id_via_ili

        vocabulary_lexicon = self.model.vocabulary.lexicon
        if (
            vocabulary_lexicon is None
            or not isinstance(dataset2, WnEntitySource)
            or dataset2.lexicon == vocabulary_lexicon
        ):
            return pairs

        translated_id_cache: dict[str, str | None] = {}
        translated_pairs = []
        for mention, sense in pairs:
            if sense.id not in translated_id_cache:
                translated_id_cache[sense.id] = synset_id_via_ili(
                    sense.id, lexicon=dataset2.lexicon, target_lexicon=vocabulary_lexicon
                )
            mapped_id = translated_id_cache[sense.id]
            if mapped_id is None:
                translated_pairs.append((mention, sense))
            else:
                translated_pairs.append((mention, replace(sense, id=mapped_id)))
        return translated_pairs

    def link(
        self,
        dataset1: list[Entity],
        dataset2: list[Entity] | EntitySource,
        graph: Graph = None,
        blocking: BlockingStrategy = DEFAULT_BLOCKING,
    ) -> list[AlignmentResult]:
        return self.matching.match(self.score_candidates(dataset1, dataset2, blocking))
