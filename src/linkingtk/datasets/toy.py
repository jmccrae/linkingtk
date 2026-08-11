"""Small, hand-curated toy datasets for quick end-to-end testing.

Unlike the other loaders in this package, these don't fetch from an
external source: they're bundled directly in the code so every algorithm
in the MVP milestone can be exercised without network access. Each one
picks classically ambiguous words/mentions with two senses/candidates
apiece, so a context-blind baseline (e.g. picking the first candidate)
would score no better than chance.
"""

from __future__ import annotations

from linkingtk.core.entity import Entity
from linkingtk.datasets.base import DatasetLoader


class ToyEADataset(DatasetLoader):
    """Three same-city entities across two knowledge graphs (6 entities).

    ``kg2``'s labels append the country name (e.g. ``"Paris, France"``), so
    an exact label match misses every pair — a blocking strategy tolerant
    of partial overlap (e.g. [LabelOverlap][linkingtk.blocking.label_overlap.LabelOverlap])
    is needed to find the candidates.
    """

    def load(self) -> tuple[list[Entity], list[Entity], list[tuple[str, str]]]:
        kg1 = [
            Entity(id="kg1:paris", labels=["Paris"]),
            Entity(id="kg1:berlin", labels=["Berlin"]),
            Entity(id="kg1:rome", labels=["Rome"]),
        ]
        kg2 = [
            Entity(id="kg2:paris", labels=["Paris, France"]),
            Entity(id="kg2:berlin", labels=["Berlin, Germany"]),
            Entity(id="kg2:rome", labels=["Rome, Italy"]),
        ]
        ground_truth = [
            ("kg1:paris", "kg2:paris"),
            ("kg1:berlin", "kg2:berlin"),
            ("kg1:rome", "kg2:rome"),
        ]
        return kg1, kg2, ground_truth


class ToyWSADataset(DatasetLoader):
    """Four ambiguous-word senses across two differently-worded dictionaries (8 entities).

    Each dictionary glosses the same four senses of *"mouse"* and *"bat"*
    in different words, so a context-blind baseline would score no better
    than chance between the two same-label candidates.
    """

    def load(self) -> tuple[list[Entity], list[Entity], list[tuple[str, str]]]:
        dict1 = [
            Entity(
                id="d1:mouse.1", labels=["mouse"], description="a small rodent with a long tail"
            ),
            Entity(
                id="d1:mouse.2",
                labels=["mouse"],
                description="a handheld pointing device for a computer",
            ),
            Entity(
                id="d1:bat.1", labels=["bat"], description="a flying mammal that is active at night"
            ),
            Entity(
                id="d1:bat.2",
                labels=["bat"],
                description="a piece of sports equipment used to hit a ball",
            ),
        ]
        dict2 = [
            Entity(
                id="d2:mouse.a",
                labels=["mouse"],
                description="small furry rodent, often a household pest",
            ),
            Entity(
                id="d2:mouse.b",
                labels=["mouse"],
                description="computer input device you move by hand",
            ),
            Entity(id="d2:bat.a", labels=["bat"], description="nocturnal flying mammal"),
            Entity(
                id="d2:bat.b",
                labels=["bat"],
                description="wooden club used to strike a ball in cricket or baseball",
            ),
        ]
        ground_truth = [
            ("d1:mouse.1", "d2:mouse.a"),
            ("d1:mouse.2", "d2:mouse.b"),
            ("d1:bat.1", "d2:bat.a"),
            ("d1:bat.2", "d2:bat.b"),
        ]
        return dict1, dict2, ground_truth


class ToyWSDDataset(DatasetLoader):
    """Four ambiguous-word mentions against two senses apiece (12 entities).

    (The classic "bank" example is deliberately left out here since it's
    already the walkthrough example in ``examples/lesk_wsd.py`` and the
    docs — this dataset covers different words so the two don't just
    duplicate each other.)
    """

    def load(self) -> tuple[list[Entity], list[Entity], list[tuple[str, str]]]:
        mentions = [
            Entity(id="m1", labels=["bass"], context="He caught a bass while fishing at dawn"),
            Entity(id="m2", labels=["bass"], context="The bass in this music track was too quiet"),
            Entity(
                id="m3", labels=["crane"], context="A crane lifted the steel beams onto the roof"
            ),
            Entity(
                id="m4",
                labels=["crane"],
                context="The crane was a large wading bird spotted near the marsh",
            ),
        ]
        senses = [
            Entity(
                id="bass.n.01",
                labels=["bass"],
                description="a species of freshwater or marine fish",
            ),
            Entity(id="bass.n.02", labels=["bass"], description="the low-frequency part in music"),
            Entity(
                id="crane.n.01",
                labels=["crane"],
                description="a machine for lifting and moving heavy objects",
            ),
            Entity(
                id="crane.n.02",
                labels=["crane"],
                description="a large wading bird with a long neck and legs",
            ),
        ]
        ground_truth = [
            ("m1", "bass.n.01"),
            ("m2", "bass.n.02"),
            ("m3", "crane.n.01"),
            ("m4", "crane.n.02"),
        ]
        return mentions, senses, ground_truth


class ToyELDataset(DatasetLoader):
    """Six ambiguous-name mentions against two knowledge-base entries apiece (18 entities)."""

    def load(self) -> tuple[list[Entity], list[Entity], list[tuple[str, str]]]:
        mentions = [
            Entity(id="m1", labels=["Paris"], context="Paris is the capital of France"),
            Entity(
                id="m2",
                labels=["Paris"],
                context="Paris, Texas is a small city in the United States",
            ),
            Entity(id="m3", labels=["Mercury"], context="Mercury is the closest planet to the Sun"),
            Entity(
                id="m4", labels=["Mercury"], context="Mercury is a liquid metal at room temperature"
            ),
            Entity(
                id="m5",
                labels=["Amazon"],
                context="The Amazon rainforest spans several South American countries",
            ),
            Entity(
                id="m6",
                labels=["Amazon"],
                context="Amazon is a technology company that sells almost everything online",
            ),
        ]
        knowledge_base = [
            Entity(
                id="Q90", labels=["Paris"], description="capital and most populous city of France"
            ),
            Entity(id="Q830149", labels=["Paris"], description="city in Texas, United States"),
            Entity(
                id="Q308",
                labels=["Mercury"],
                description="first planet from the Sun in the Solar System",
            ),
            Entity(
                id="Q925",
                labels=["Mercury"],
                description="chemical element, a heavy silvery liquid metal",
            ),
            Entity(
                id="Q3733",
                labels=["Amazon"],
                description="moist broadleaf tropical forest region in South America",
            ),
            Entity(
                id="Q3884",
                labels=["Amazon"],
                description="American multinational technology company",
            ),
        ]
        ground_truth = [
            ("m1", "Q90"),
            ("m2", "Q830149"),
            ("m3", "Q308"),
            ("m4", "Q925"),
            ("m5", "Q3733"),
            ("m6", "Q3884"),
        ]
        return mentions, knowledge_base, ground_truth
