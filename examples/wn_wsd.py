"""Word Sense Disambiguation against live WordNet, via `wn`.

Disambiguates the classic ambiguous word "bank" between its financial and
riverbank senses, exactly like `lesk_wsd.py`, but scoring against every
sense `wn` knows about instead of two hand-picked candidates --
`WnEntitySource` fetches only the ones `ExactMatch` actually asks for
(`bank`), rather than materializing the whole lexicon as `Entity` objects.

Requires the `wn` optional dependency and a downloaded lexicon:

    uv pip install linkingtk[wn]
    python -m wn download oewn:2021

Run with: `uv run python examples/wn_wsd.py`
"""

from linkingtk.algorithms.wsd import LeskLinker
from linkingtk.blocking import ExactMatch
from linkingtk.core import Entity
from linkingtk.sources import WnEntitySource


def main() -> None:
    mentions = [
        Entity(
            id="m1",
            labels=["bank"],
            context="The bank approved my loan application for a new lending institution",
        ),
    ]
    senses = WnEntitySource(lang="en")

    results = LeskLinker().link(mentions, senses, blocking=ExactMatch())
    for result in results:
        print(f"{result.source_id} -> {result.target_id} (score={result.score})")
        print(f"  alternatives: {result.alternatives}")

    target = senses.get(results[0].target_id)
    assert target is not None
    print(f"  gloss: {target.description}")


if __name__ == "__main__":
    main()
