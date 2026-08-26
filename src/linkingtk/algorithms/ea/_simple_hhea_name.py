"""ALBERT name-embedding + whitening for
[SimpleHHEALinker][linkingtk.algorithms.ea.simple_hhea.SimpleHHEALinker].

Ports `process_name_embedding.py` from
https://github.com/DataArcTech/Simple-HHEA: mean-pooled ALBERT embeddings
of each entity's label text, followed by whitening (Su et al. 2021,
"Whitening Sentence Representations for Better Semantics and Faster
Retrieval") -- a linear transform derived from the SVD of the *combined*
covariance of both KGs' raw embeddings that reduces dimensionality and
improves cosine-similarity behavior of BERT-family sentence embeddings.
`transformers`/`torch` are core dependencies already, so this needs no
`OptionalDependencyError` guard (unlike the structural-embedding stage's
new `node2vec` dependency).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from linkingtk.core.entity import Entity, label_texts
from linkingtk.utils.device import resolve_device

if TYPE_CHECKING:
    import numpy.typing as npt


def entity_name_text(entity: Entity) -> str:
    """Cleaned label text for name-embedding, per the reference's own
    ``name.split("/")[-1].replace("_", " ")`` (strips a URI down to its
    trailing local name, then de-underscores it)."""
    label = label_texts(entity)[0] if entity.labels else entity.id
    return label.split("/")[-1].replace("_", " ").replace("\xa0", "")


def embed_names(
    entities: list[Entity],
    model_name: str = "albert-base-v2",
    device: str = "cpu",
    batch_size: int = 64,
) -> npt.NDArray[np.floating[Any]]:
    """Mean-pooled ALBERT embedding for each entity's cleaned label text.

    Returns a raw ``(len(entities), hidden_size)`` array -- not whitened.
    [compute_kernel_bias][linkingtk.algorithms.ea._simple_hhea_name.compute_kernel_bias]/
    [whiten][linkingtk.algorithms.ea._simple_hhea_name.whiten] must be fit
    jointly over *both* KGs' raw embeddings (matching the reference's own
    combined-covariance whitening), so aren't folded in here.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    resolved = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(resolved)
    model.eval()

    texts = [entity_name_text(entity) for entity in entities]
    chunks: list[npt.NDArray[np.floating[Any]]] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(resolved)
            hidden_states = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            chunks.append(pooled.cpu().numpy())
    if not chunks:
        hidden_size: int = model.config.hidden_size
        return np.empty((0, hidden_size), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def compute_kernel_bias(
    vectors: npt.NDArray[np.floating[Any]], n_components: int
) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
    """Whitening kernel/bias from ``vectors``' covariance.

    Ports `process_name_embedding.py`'s ``compute_kernel_bias`` -- the
    final transform is ``(x + bias) @ kernel``.
    """
    mean = vectors.mean(axis=0, keepdims=True)
    covariance = np.cov(vectors.T)
    u, singular_values, _ = np.linalg.svd(covariance)
    kernel = u @ np.diag(1 / np.sqrt(singular_values))
    return kernel[:, :n_components], -mean


def whiten(
    vectors: npt.NDArray[np.floating[Any]],
    kernel: npt.NDArray[np.floating[Any]],
    bias: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """Applies a whitening transform and L2-normalizes each row.

    Ports `process_name_embedding.py`'s ``transform_and_normalize``.
    """
    transformed = (vectors + bias) @ kernel
    norms = np.linalg.norm(transformed, axis=1, keepdims=True)
    result: npt.NDArray[np.floating[Any]] = transformed / np.clip(norms, 1e-12, None)
    return result
