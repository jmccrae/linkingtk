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
