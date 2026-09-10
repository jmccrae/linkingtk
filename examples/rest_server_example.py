"""Offer a linker as a REST service, wrapping BaseLinker.link.

Run with: `uv run python examples/rest_server_example.py`, then e.g.:

    curl -X POST localhost:8000/link/exact-match \\
      -F 'request={"dataset1":[{"id":"s1","labels":["cat"]}],
                    "dataset2_entities":[{"id":"t1","labels":["cat"]},
                                          {"id":"t2","labels":["dog"]}]}'

`app` is also importable directly for production-style deployment, e.g.
`uvicorn examples.rest_server_example:app`.
"""

from linkingtk.algorithms.base import DEFAULT_BLOCKING, BaseLinker
from linkingtk.core.result import AlignmentResult
from linkingtk.server import create_app


class ExactMatchLinker(BaseLinker):
    """Turns ExactMatch's candidate pairs directly into links, for demo purposes."""

    def link(self, dataset1, dataset2, graph=None, blocking=DEFAULT_BLOCKING):
        pairs = blocking.candidate_pairs(dataset1, dataset2)
        return [AlignmentResult(source_id=a.id, target_id=b.id) for a, b in pairs]


# A pre-registered EntitySource can be referenced by name too, instead of
# sending dataset2 inline -- see docs/examples/rest_server.md.
app = create_app(linkers={"exact-match": ExactMatchLinker()})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
