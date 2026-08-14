"""Pretrained-transformer literal encoder for
[MultiKELinker][linkingtk.algorithms.ea.multike.MultiKELinker].

Replaces OpenEA's own ``name_embeds``/``literal_embeds`` pipeline
(``src/openea/approaches/literal_encoder.py``: a pretrained English-only
word2vec/fastText file plus a custom stacked autoencoder trained via
reconstruction loss) with a small pretrained **multilingual** transformer
(``transformers`` is already a hard dependency of this repo -- zero new
install), mean-pooled and projected down to ``embedding_dim`` via a fixed
random orthogonal projection. No training loop is needed here (unlike
OpenEA's autoencoder) because the transformer already produces
well-formed semantic structure; see
[MultiKELinker][linkingtk.algorithms.ea.multike.MultiKELinker]'s module
docstring for the full rationale -- this is the one place this port
genuinely departs from a faithful reimplementation, rather than a
tractability workaround.

Kept separate from ``_multike_torch.py`` and imported lazily (inside
``encode_literals`` itself, matching every other optional-dependency
import in this package) so importing ``linkingtk.algorithms.ea`` never
requires ``transformers``' heavier runtime dependencies (or a model
download) unless a MultiKE ``fit()`` actually runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


def encode_literals(
    texts: list[str],
    model_name: str,
    embedding_dim: int,
    max_length: int = 16,
    batch_size: int = 256,
    random_state: int | None = None,
    device: Any = "cpu",
) -> npt.NDArray[np.float32]:
    """Encode ``texts`` into ``(len(texts), embedding_dim)`` L2-normalized vectors.

    Each text is tokenized/truncated to ``max_length`` tokens, run through
    a frozen (``eval()``, ``no_grad()``) pretrained transformer in
    batches of ``batch_size``, mean-pooled over the attention mask (so
    padding tokens don't dilute the average), L2-normalized, then
    projected down to ``embedding_dim`` by a fixed random orthogonal
    matrix (seeded from ``random_state``) and L2-normalized again. The
    projection is never trained -- it exists only to match
    ``embedding_dim``, not to learn anything (unlike OpenEA's own
    stacked-autoencoder compression step, which this replaces).

    Returns a plain numpy array aligned 1:1 with ``texts`` -- callers
    build their own ``text -> row index`` mapping.

    Args:
        device: Torch device the model and every batch run on -- this is
            a batched transformer forward pass over potentially thousands
            of names/values, the single most GPU-friendly step in
            MultiKE's training pipeline. The returned array is always a
            plain CPU numpy array regardless of this setting.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not texts:
        return np.empty((0, embedding_dim), dtype=np.float32)

    if random_state is not None:
        torch.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    pooled_batches: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            output = model(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            summed = (output.last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            pooled_batches.append(summed / counts)

        pooled = torch.nn.functional.normalize(torch.cat(pooled_batches, dim=0), dim=1)
        hidden_size = pooled.shape[1]
        projection = torch.nn.init.orthogonal_(
            torch.empty(hidden_size, embedding_dim, device=device)
        )
        projected = torch.nn.functional.normalize(pooled @ projection, dim=1)

    result: npt.NDArray[np.float32] = projected.cpu().numpy().astype(np.float32)
    return result
