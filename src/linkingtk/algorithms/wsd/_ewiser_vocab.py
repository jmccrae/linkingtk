"""Sense vocabulary: index <-> synset id, for EWISER's full-inventory output layer.

[EwiserEncoder][linkingtk.algorithms.wsd.ewiser.EwiserEncoder] classifies
every word against a fixed, closed vocabulary of WordNet senses rather than
scoring one candidate at a time (unlike
[GlossBertEncoder][linkingtk.algorithms.wsd.glossbert.GlossBertEncoder]'s
cross-encoder). `SenseVocabulary` is that fixed index space.

Two construction paths, since the index order matters differently
depending on where it comes from:

- `from_offsets_file` loads EWISER's own published checkpoint dictionary
  (frequency-sorted over the original authors' training corpus) -- this
  order is **not** reproducible from first principles (it depends on
  corpus statistics we don't have), so checkpoint-compatible inference
  requires the literal file, supplied locally by the caller (mirrors how
  `examples/glossbert_reproduction.py` requires a local copy of GlossBERT's
  own published checkpoint directory -- neither file is redistributed by
  this package).
- `from_wn` builds a fresh vocabulary for from-scratch training, where any
  consistent order works -- nothing downstream depends on matching
  upstream's specific index assignment.
"""

from __future__ import annotations

from pathlib import Path

from linkingtk.sources.wn import wn30_offset_to_synset_id


class SenseVocabulary:
    """index <-> synset id mapping for an `EwiserEncoder`'s output layer.

    Args:
        index_to_synset_id: The vocabulary itself, in index order. An entry
            of ``None`` marks a reserved/special slot (e.g. fairseq's
            ``<s>``/``<pad>``/``</s>``/``<unk>``, or an offset that didn't
            resolve in the target lexicon) that never matches a real
            candidate sense.
    """

    def __init__(self, index_to_synset_id: list[str | None]) -> None:
        self._index_to_synset_id = index_to_synset_id
        self._synset_id_to_index = {
            synset_id: index
            for index, synset_id in enumerate(index_to_synset_id)
            if synset_id is not None
        }

    @classmethod
    def from_offsets_file(
        cls, path: str | Path, lexicon: str = "omw-en:1.4", nspecial: int = 4
    ) -> SenseVocabulary:
        """Load EWISER's own ``res/dictionaries/offsets.txt``.

        Each line is ``"<token> <frequency>"``; `token` is either a
        WordNet 3.0 offset (``"wn:02084071n"``) or a non-offset dummy
        symbol (e.g. fairseq's ``"madeupword0000"`` padding entry), which
        becomes a reserved (``None``) slot like the `nspecial` prefix.
        Offsets that don't resolve in `lexicon` (version skew) also become
        ``None`` rather than raising, since a handful of unresolvable
        entries shouldn't block loading the other ~117,000.

        Args:
            path: Local path to `offsets.txt` (not bundled with this
                package -- see the module docstring).
            lexicon: Passed to
                [wn30_offset_to_synset_id][linkingtk.sources.wn.wn30_offset_to_synset_id].
            nspecial: Reserved slots prepended before the file's own
                entries, matching fairseq's convention (``<s>``, ``<pad>``,
                ``</s>``, ``<unk>``) the checkpoints were trained with.

        Returns:
            A `SenseVocabulary` of length ``nspecial + <lines in path>``,
            matching a published checkpoint's ``decoder.logits.weight``
            row count (117664 for the three EWISER checkpoints released
            alongside the paper).
        """
        index_to_synset_id: list[str | None] = [None] * nspecial
        for line in Path(path).read_text().splitlines():
            token = line.split(maxsplit=1)[0] if line.strip() else ""
            if token.startswith("wn:"):
                index_to_synset_id.append(wn30_offset_to_synset_id(token, lexicon=lexicon))
            else:
                index_to_synset_id.append(None)
        return cls(index_to_synset_id)

    @classmethod
    def from_wn(cls, synset_ids: list[str], nspecial: int = 4) -> SenseVocabulary:
        """Build a fresh vocabulary for from-scratch training.

        Args:
            synset_ids: The senses to index, e.g. every synset observed in
                a training corpus. Deduplicated and sorted for a
                deterministic (order-independent-correctness) index
                assignment.
            nspecial: Reserved slots prepended before the real entries.

        Returns:
            A `SenseVocabulary` of length ``nspecial + len(set(synset_ids))``.
        """
        index_to_synset_id: list[str | None] = [None] * nspecial
        index_to_synset_id.extend(sorted(set(synset_ids)))
        return cls(index_to_synset_id)

    def index_for(self, synset_id: str) -> int | None:
        """The output-layer index for `synset_id`, or ``None`` if it's not in this vocabulary."""
        return self._synset_id_to_index.get(synset_id)

    def synset_id_for(self, index: int) -> str | None:
        """The synset id at `index`, or ``None`` for a reserved/unresolved slot."""
        return self._index_to_synset_id[index]

    def __len__(self) -> int:
        return len(self._index_to_synset_id)
