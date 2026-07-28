from linkingtk.core.entity import Entity
from linkingtk.core.result import AlignmentResult


def test_entity_minimal() -> None:
    entity = Entity(id="e1", labels=["cat"])
    assert entity.id == "e1"
    assert entity.labels == ["cat"]
    assert entity.description is None
    assert entity.context is None
    assert entity.properties == {}


def test_entity_with_language_tagged_fields() -> None:
    entity = Entity(
        id="e2",
        labels=[("cat", "en"), ("chat", "fr")],
        description=("a small domesticated carnivore", "en"),
        context=("I saw a cat today", 9, 12),
        properties={"pos": "noun"},
    )
    assert entity.labels[0] == ("cat", "en")
    assert entity.context == ("I saw a cat today", 9, 12)
    assert entity.properties["pos"] == "noun"


def test_alignment_result_defaults() -> None:
    result = AlignmentResult(source_id="e1", target_id="e2")
    assert result.score == 1.0
    assert result.alternatives == []
