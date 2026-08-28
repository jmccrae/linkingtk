from linkingtk.algorithms.wsd import LeskLinker
from linkingtk.core.entity import Entity


def test_lesk_picks_sense_with_more_gloss_overlap() -> None:
    mentions = [
        Entity(id="m1", labels=["bank"], context="I deposited money at the bank yesterday"),
    ]
    senses = [
        Entity(
            id="bank.n.01",
            labels=["bank"],
            description="a financial institution that accepts deposits of money",
        ),
        Entity(
            id="bank.n.02",
            labels=["bank"],
            description="the land alongside a river",
        ),
    ]

    results = LeskLinker().link(mentions, senses)

    assert len(results) == 1
    result = results[0]
    assert result.source_id == "m1"
    assert result.target_id == "bank.n.01"
    assert result.alternatives == ["bank.n.02"]
    assert result.score > 0


def test_lesk_only_considers_candidates_from_blocking() -> None:
    mentions = [Entity(id="m1", labels=["bank"], context="the money in the bank")]
    senses = [Entity(id="s1", labels=["river"], description="a natural flowing watercourse")]

    results = LeskLinker().link(mentions, senses)

    assert results == []


def test_lesk_falls_back_to_first_listed_sense_on_zero_overlap() -> None:
    # No context word overlaps any gloss, so every candidate ties at
    # score 0.0. The fallback should be the most-frequent sense -- i.e.
    # whichever candidate the blocking step (and, for a real WnEntitySource,
    # the lexicon's own sense order) listed first -- not whichever
    # target_id happens to sort first alphabetically. "zzz.n.01" is
    # deliberately last alphabetically to prove this (issue #66).
    mentions = [Entity(id="m1", labels=["tree"], context="he was a famous actor")]
    senses = [
        Entity(id="zzz.n.01", labels=["tree"], description="a woody perennial plant"),
        Entity(id="aaa.n.02", labels=["tree"], description="a surname"),
    ]

    results = LeskLinker().link(mentions, senses)

    assert results[0].target_id == "zzz.n.01"
    assert results[0].score == 0.0


def test_lesk_handles_multiple_mentions_independently() -> None:
    mentions = [
        Entity(id="m1", labels=["bank"], context="deposited cash at the bank"),
        Entity(id="m2", labels=["bank"], context="fishing along the river bank"),
    ]
    senses = [
        Entity(id="bank.n.01", labels=["bank"], description="a financial institution"),
        Entity(id="bank.n.02", labels=["bank"], description="the land alongside a river"),
    ]

    results = {r.source_id: r for r in LeskLinker().link(mentions, senses)}

    assert results["m1"].target_id == "bank.n.01"
    assert results["m2"].target_id == "bank.n.02"
