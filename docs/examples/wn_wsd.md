# WSD against live WordNet with `wn`

The same "bank" disambiguation as [Lesk WSD](lesk_wsd.md), but the candidate
side is [`WnEntitySource`](../reference/sources.md), a query-driven
[`EntitySource`](../reference/core.md) wrapping the
[`wn`](https://github.com/goodmami/wn) library. Instead of two hand-picked
candidate senses, [`ExactMatch`](../reference/blocking.md) queries `wn` for
every synset lexicalized as `"bank"` -- the full ambiguity set a real
dictionary has, not a toy subset -- without ever materializing the whole
lexicon as `Entity` objects.

Requires the `wn` optional dependency and a one-time lexicon download:

```bash
uv pip install linkingtk[wn]
python -m wn download oewn:2021
```

```python
--8<-- "examples/wn_wsd.py"
```

Run with:

```bash
uv run python examples/wn_wsd.py
```

```text
m1 -> oewn-08437235-n (score=2.0)
  alternatives: ['oewn-00170126-n', 'oewn-02790795-n', 'oewn-04146942-n', 'oewn-08479077-n', 'oewn-09236341-n', 'oewn-09236472-n', 'oewn-09236735-n', 'oewn-13377435-n', 'oewn-13389491-n']
  gloss: a financial institution that accepts deposits and channels the money into lending activities
```
