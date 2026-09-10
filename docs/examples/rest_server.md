# REST server mode

[`create_app`](../reference/server.md) wraps any [`BaseLinker`](../reference/algorithms.md)
as a REST service, mainly by exposing
[`BaseLinker.link`](../reference/algorithms.md) as `POST /link/{linker_name}`.
The deployer constructs whatever linker(s)/[`EntitySource`](../reference/core.md)(s)
they need in Python -- however they need to (model paths, device,
checkpoints, API keys, ...) -- and registers them by name; there's no
generic "build a linker from JSON config" layer, since the ~25 linkers in
this library have very different constructors.

```python
--8<-- "examples/rest_server_example.py"
```

Run with:

```bash
uv run python examples/rest_server_example.py
```

or, for production-style deployment where `uvicorn` manages the process:

```bash
uvicorn examples.rest_server_example:app
```

Requires the `server` extra: `uv sync --extra server` (or `pip install
"linkingtk[server]"`).

## Endpoints

- `GET /health` -- liveness check.
- `GET /linkers` -- names of the registered linkers.
- `GET /sources` -- names of the registered entity sources.
- `POST /link/{linker_name}` -- runs `link()`. A `multipart/form-data`
  request with:
    - `request` (required): a JSON object with `dataset1` (a list of
      entities) and *exactly one* of `dataset2_entities` (a list of
      entities, sent inline) or `dataset2_source` (the name of a
      pre-registered `EntitySource`, passed straight through to `link()` so
      search-based blocking still works instead of materializing every
      candidate).
    - `graph` (optional): an uploaded file with the supporting graph.
    - `graph_format` (optional, default `"triples"`): `"triples"` for a
      JSON list of `[subject, predicate, object]` triples, `"rdf"` for any
      RDF serialization `rdflib` can parse (Turtle, N-Triples, RDF/XML,
      JSON-LD, ...), or `"graphml"` / `"gexf"` / `"edgelist"` for the
      matching NetworkX file format.

Response: a JSON list of `{"source_id", "target_id", "score",
"alternatives"}` objects, mirroring
[`AlignmentResult`](../reference/core.md).

## Targeting an EntitySource by name

Pass a `sources` dict to `create_app` alongside `linkers` to let requests
reference a live source (Wikidata, a local vector index, ...) instead of
sending `dataset2` inline:

```python
app = create_app(
    linkers={"exact-match": ExactMatchLinker()},
    sources={"wikidata": WikidataEntitySource()},
)
```

```bash
curl -X POST localhost:8000/link/exact-match \
  -F 'request={"dataset1":[{"id":"s1","labels":["cat"]}],"dataset2_source":"wikidata"}'
```

## Uploading a graph

```bash
curl -X POST localhost:8000/link/my-linker \
  -F 'request={"dataset1":[...],"dataset2_entities":[...]}' \
  -F 'graph_format=rdf' \
  -F 'graph=@my_graph.ttl'
```

Blocking-strategy selection isn't exposed over REST yet -- each linker
always runs with its own default blocking strategy.
