# LLM linker benchmarks (EA/EL/WSD)

`LlmBaseLinker` scored against one small real dataset per task: the OAEI
`conference` ontology-matching pair (EA), MSNBC (EL), and SemEval-2007 (WSD).
Unlike this repo's other benchmarks, every prediction here is a real LLM
call, so EL/WSD are capped to 30 source entities by default -- see the
script's own docstring for why, and how to run the uncapped split.

```python
--8<-- "examples/llm_benchmark.py"
```

Run with:

```bash
uv run python examples/llm_benchmark.py
```

Against a local `llama3.2:3b` (Ollama):

```text
EA: {'Hits@1': 0.652, 'Hits@5': 0.826, 'MRR': 0.732} (31.7s)
EL: {'precision@1': 0.9, 'recall': 0.9, 'f1': 0.9} (21.5s)
WSD: {'precision@1': 0.464, 'recall': 0.433, 'f1': 0.448} (49.0s)
```

Two real findings from getting these numbers, not just the metrics
themselves:

- A first attempt with the even smaller `llama3.2:1b` was unusable --
  it frequently returned its own source entity's id as a "candidate"
  rather than a real one. `LlmBaseLinker` logged and skipped every one of
  these rather than crashing or silently mis-scoring, which is exactly
  the point of that defensive path, but the resulting benchmark was
  meaningless. `llama3.2:3b` was reliable enough for real numbers.
- `ExactMatch` (the default EL blocking strategy) gives at
  least one candidate to only 291 of MSNBC's 747 mentions (39%) --
  measured directly, not assumed -- because mention surface forms in news
  text often don't exactly match their KB entity's canonical Wikipedia
  title. This script uses `LabelOverlap` for EL instead; using
  `ExactMatch` would have silently scored `LlmBaseLinker` on well under
  half the dataset and made it look far worse than it is.

Also fixed two real bugs in `LlmBaseLinker` itself while producing this
benchmark (both already covered by regression tests in
`tests/algorithms/test_llm.py`):

- Some models echo the prompt's own `"(id=...)"` candidate rendering back
  verbatim instead of the bare id (e.g. `"id=Atlanta_Falcons"`) --
  `LlmBaseLinker` now recovers the real id from that instead of discarding
  it as hallucinated.
- The default `max_tokens` (1024) was too tight for a source entity with
  many real candidates -- raised to 2048. WSD's own wide `top_k=50`
  blocking still needed a further per-call override to 4096 (see the
  script's `run_wsd` for why this is a call-site tuning knob, not a
  library-wide default).
