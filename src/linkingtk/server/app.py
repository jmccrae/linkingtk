"""FastAPI app factory wrapping [BaseLinker.link][linkingtk.algorithms.base.BaseLinker.link]
as a REST endpoint.

The deployer constructs whatever [BaseLinker][linkingtk.algorithms.base.BaseLinker]
and [EntitySource][linkingtk.core.source.EntitySource] instances they need
(model paths, device, checkpoints, API keys, ...) and registers them by name
via [create_app][linkingtk.server.app.create_app]; the returned app is run
the normal way, e.g. ``uvicorn mymodule:app``.
"""

from typing import TYPE_CHECKING

from linkingtk.algorithms.base import BaseLinker
from linkingtk.core.entity import Entity
from linkingtk.core.source import EntitySource
from linkingtk.exceptions import OptionalDependencyError
from linkingtk.server.graph_io import parse_graph
from linkingtk.server.schemas import LinkRequestBody, SchemaAlignmentResult

if TYPE_CHECKING:
    from fastapi import FastAPI


def create_app(
    linkers: dict[str, BaseLinker],
    sources: dict[str, EntitySource] | None = None,
) -> "FastAPI":
    """Build a FastAPI app exposing the given linkers/sources over REST.

    Args:
        linkers: Maps a name (used in the ``POST /link/{name}`` URL) to a
            ready-to-use `BaseLinker` instance.
        sources: Maps a name to a ready-to-use `EntitySource`, so it can be
            referenced by name as a request's `dataset2` instead of
            materializing entities inline. Passed through to `link()`
            as-is, preserving search-based blocking behavior.

    Returns:
        A FastAPI app with `GET /health`, `GET /linkers`, `GET /sources`,
        and `POST /link/{linker_name}` routes.

    Raises:
        OptionalDependencyError: If `fastapi` isn't installed.
    """
    try:
        from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    except ImportError as exc:
        raise OptionalDependencyError("REST server", "server") from exc

    sources = sources or {}
    app = FastAPI(title="linkingtk REST server")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/linkers")
    def list_linkers() -> list[str]:
        return sorted(linkers)

    @app.get("/sources")
    def list_sources() -> list[str]:
        return sorted(sources)

    @app.post("/link/{linker_name}")
    def link(
        linker_name: str,
        request: str = Form(...),
        graph: UploadFile | None = File(None),  # noqa: B008
        graph_format: str = Form("triples"),
    ) -> list[SchemaAlignmentResult]:
        linker = linkers.get(linker_name)
        if linker is None:
            raise HTTPException(status_code=404, detail=f"Unknown linker {linker_name!r}")

        try:
            body = LinkRequestBody.model_validate_json(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        dataset1 = [e.to_entity() for e in body.dataset1]
        dataset2: list[Entity] | EntitySource
        if body.dataset2_entities is not None:
            dataset2 = [e.to_entity() for e in body.dataset2_entities]
        else:
            assert body.dataset2_source is not None  # enforced by LinkRequestBody validator
            source = sources.get(body.dataset2_source)
            if source is None:
                raise HTTPException(
                    status_code=404, detail=f"Unknown source {body.dataset2_source!r}"
                )
            dataset2 = source

        parsed_graph = None
        if graph is not None:
            try:
                parsed_graph = parse_graph(graph.file.read(), graph_format)
            except (ValueError, OptionalDependencyError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        results = linker.link(dataset1, dataset2, parsed_graph)
        return [SchemaAlignmentResult.from_result(r) for r in results]

    return app
