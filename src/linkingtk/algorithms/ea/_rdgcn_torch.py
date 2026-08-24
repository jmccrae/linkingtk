"""PyTorch model and training-step functions for
[RDGCNLinker][linkingtk.algorithms.ea.rdgcn.RDGCNLinker].

Ports OpenEA's ``Layer``/``RDGCN`` (``approaches/rdgcn.py``): two rounds of
dual-graph (relation) attention projected onto the primal (entity) graph via
sparse attention, each mixed back into the running entity embedding by a
learned scalar (``alpha``/``beta``), followed by two diagonal-weight GCN
layers each wrapped in a highway gate. Uses plain ``torch.sparse``
throughout (including ``torch.sparse.softmax``, PyTorch's own built-in
sparse-row-softmax primitive) -- no `torch_geometric`/`dgl` dependency, same
posture as ``_gcn_align_torch.py``.

Callers must already have confirmed ``torch`` is importable
(``rdgcn.py``'s ``fit()`` does this via ``OptionalDependencyError`` before
any of these run) -- same precedent as ``_gcn_align_torch.py``/
``_kdcoe_torch.py``.

**Dimension convention** (confirmed by reading ``rdgcn.py``'s ``Layer.build()``
directly): entity/primal embeddings have dimensionality ``D`` (the
constructor's ``embedding_dim``, which must equal the pretrained word
vectors' own dimensionality -- no projection layer exists between them).
Per-relation "dual graph" features are ``2 * D`` (``concat([mean head-entity
embeds, mean tail-entity embeds])``) -- every dense-attention layer's query
projection takes ``2 * D`` in and produces ``D`` out; the sparse attention
layer's relation-score projection takes ``2 * D`` in and produces a single
scalar per edge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def build_rdgcn_model(
    initial_primal_embeddings: torch.Tensor,
    embedding_dim: int,
    alpha: float,
    beta: float,
) -> torch.nn.Module:
    """Ports ``Layer.build()``'s computation graph as a live PyTorch model.

    A factory function rather than a module-level ``torch.nn.Module``
    subclass, since ``torch.nn.Module`` can't be named at module scope
    without ``torch`` installed -- same precedent as
    ``_gcn_align_torch.build_gcn_align_branch``/
    ``_kdcoe_torch.build_description_encoder``.

    Unlike
    [build_gcn_align_branch][linkingtk.algorithms.ea._gcn_align_torch.build_gcn_align_branch],
    the returned model's ``forward()`` takes the fixed graph structures
    (adjacencies/masks/edge lists) as *arguments* rather than storing them
    internally -- they never change during training (RDGCN has no
    AliNet-style bootstrapping adjacency mutation), so this just keeps
    device-placement simple: build every fixed tensor directly on the
    target device before constructing the model, same as
    ``gcn_align.py``'s own ``fit()`` does for its adjacency.

    Confirmed from ``Layer.build()`` directly: OpenEA's own
    ``get_pretrained_input`` wraps the name-embedding init in a plain
    ``tf.Variable`` (trainable by default, no ``trainable=False``) --
    ``primal_x0`` is a **trained** parameter seeded from the pretrained
    name embeddings, not a frozen input. This is why the per-relation
    "dual graph" features (which depend on the live primal embeddings)
    must be recomputed every forward pass, not cached once.

    Args:
        initial_primal_embeddings: ``(num_entities, embedding_dim)`` --
            the entity-embedding parameter's initial value, e.g. from
            [init_name_embeddings][linkingtk.algorithms.ea._rdgcn_training.init_name_embeddings],
            already placed on the target device.
        embedding_dim: Entity-embedding dimensionality (``D``). Must equal
            ``initial_primal_embeddings``'s own last dimension.
        alpha: Round-1 mixing weight (``primal_X_1 = primal_X_0 + alpha *
            primal_H_1``). OpenEA's published value is ``0.1``.
        beta: Round-2 mixing weight. OpenEA's published value is ``0.3``.

    Returns:
        A module whose ``forward(primal_adjacency, dual_adjacency,
        head_mask, tail_mask, edge_heads, edge_tails, edge_relations)``
        returns ``(num_entities, embedding_dim)`` entity embeddings --
        see each argument's shape in
        [RDGCNLinker.fit][linkingtk.algorithms.ea.rdgcn.RDGCNLinker.fit]'s
        call site.
    """
    import torch
    import torch.nn.functional as functional
    from torch import nn

    dual_dim = 2 * embedding_dim

    class _DenseGraphAttentionLayer(nn.Module):
        """Ports ``add_self_att_layer``/``add_dual_att_layer`` (identical math,
        differing only in whether the query and value inputs coincide, and
        whether the query projection has a bias)."""

        def __init__(self, use_bias: bool) -> None:
            super().__init__()
            self.proj = nn.Linear(dual_dim, embedding_dim, bias=use_bias)
            self.score1 = nn.Linear(embedding_dim, 1)
            self.score2 = nn.Linear(embedding_dim, 1)

        def forward(
            self, query_input: torch.Tensor, value_input: torch.Tensor, adjacency: torch.Tensor
        ) -> torch.Tensor:
            feats = self.proj(query_input)
            f1 = self.score1(feats)
            f2 = self.score2(feats)
            logits = f1 + f2.transpose(0, 1)
            logits = adjacency * logits
            bias_mat = -1e9 * (1.0 - (adjacency > 0).float())
            coefs = torch.softmax(functional.leaky_relu(logits) + bias_mat, dim=1)
            return torch.relu(coefs @ value_input)

    class _SparseAttentionLayer(nn.Module):
        """Ports ``add_sparse_att_layer``: projects the dual-graph output to a
        per-relation scalar, gathers it onto every primal edge via
        ``edge_relations``, softmax-normalizes per source entity (row), and
        aggregates the primal layer through the result.

        Duplicate ``(head, tail)`` edges connected by *different* relations
        (the same entity pair linked by more than one relation) are summed
        by ``.coalesce()`` before the softmax, rather than kept as
        separate competing entries the way TensorFlow's uncoalesced sparse
        tensor lets OpenEA's own code do -- a deliberate simplification
        (`torch.sparse.softmax` needs a coalesced input; replicating
        per-edge-not-per-cell softmax without `torch_scatter` isn't worth
        it for what's a rare case in practice, one entity pair connected by
        multiple distinct relations simultaneously).
        """

        def __init__(self) -> None:
            super().__init__()
            self.transform = nn.Linear(dual_dim, 1)

        def forward(
            self,
            dual_layer: torch.Tensor,
            primal_layer: torch.Tensor,
            edge_heads: torch.Tensor,
            edge_tails: torch.Tensor,
            edge_relations: torch.Tensor,
            num_entities: int,
        ) -> torch.Tensor:
            relation_scores = self.transform(dual_layer).squeeze(-1)
            edge_logits = functional.leaky_relu(relation_scores[edge_relations])
            indices = torch.stack([edge_heads, edge_tails])
            sparse_logits = torch.sparse_coo_tensor(
                indices, edge_logits, size=(num_entities, num_entities)
            ).coalesce()
            coefs = torch.sparse.softmax(sparse_logits, dim=1)
            return torch.relu(torch.sparse.mm(coefs, primal_layer))

    class _DiagonalGCNLayer(nn.Module):
        """Ports ``add_diag_layer``: an elementwise (not full-matrix) learnable
        weight, propagated through the primal adjacency."""

        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.ones(1, embedding_dim))

        def forward(self, inputs: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
            return torch.relu(torch.sparse.mm(adjacency, inputs * self.scale))

    class _HighwayGate(nn.Module):
        """Ports ``highway``."""

        def __init__(self) -> None:
            super().__init__()
            self.gate = nn.Linear(embedding_dim, embedding_dim)

        def forward(self, layer1: torch.Tensor, layer2: torch.Tensor) -> torch.Tensor:
            transform_gate = torch.sigmoid(self.gate(layer1))
            carry_gate = 1.0 - transform_gate
            return transform_gate * layer2 + carry_gate * layer1

    class _RDGCNModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.primal_x0 = nn.Parameter(initial_primal_embeddings.clone())
            self.round1_att = _DenseGraphAttentionLayer(use_bias=False)
            self.round1_sparse_att = _SparseAttentionLayer()
            self.round2_att = _DenseGraphAttentionLayer(use_bias=True)
            self.round2_sparse_att = _SparseAttentionLayer()
            self.diag1 = _DiagonalGCNLayer()
            self.diag2 = _DiagonalGCNLayer()
            self.highway1 = _HighwayGate()
            self.highway2 = _HighwayGate()
            self.alpha = alpha
            self.beta = beta

        def _relation_features(
            self, primal_embeds: torch.Tensor, head_mask: torch.Tensor, tail_mask: torch.Tensor
        ) -> torch.Tensor:
            head_means = torch.sparse.mm(head_mask, primal_embeds)
            tail_means = torch.sparse.mm(tail_mask, primal_embeds)
            return torch.cat([head_means, tail_means], dim=-1)

        def forward(
            self,
            primal_adjacency: torch.Tensor,
            dual_adjacency: torch.Tensor,
            head_mask: torch.Tensor,
            tail_mask: torch.Tensor,
            edge_heads: torch.Tensor,
            edge_tails: torch.Tensor,
            edge_relations: torch.Tensor,
        ) -> torch.Tensor:
            num_entities = self.primal_x0.shape[0]
            primal_x0 = self.primal_x0

            dual_x1 = self._relation_features(primal_x0, head_mask, tail_mask)
            dual_h1 = self.round1_att(dual_x1, dual_x1, dual_adjacency)
            primal_h1 = self.round1_sparse_att(
                dual_h1, primal_x0, edge_heads, edge_tails, edge_relations, num_entities
            )
            primal_x1 = primal_x0 + self.alpha * primal_h1

            dual_x2 = self._relation_features(primal_x1, head_mask, tail_mask)
            dual_h2 = self.round2_att(dual_x2, dual_h1, dual_adjacency)
            primal_h2 = self.round2_sparse_att(
                dual_h2, primal_x1, edge_heads, edge_tails, edge_relations, num_entities
            )
            primal_x2 = primal_x0 + self.beta * primal_h2

            gcn1 = self.diag1(primal_x2, primal_adjacency)
            gcn1 = self.highway1(primal_x2, gcn1)
            gcn2 = self.diag2(gcn1, primal_adjacency)
            output: torch.Tensor = self.highway2(gcn1, gcn2)
            return output

    return _RDGCNModel()


def mine_hard_negatives(
    embeddings: torch.Tensor,
    pos_left: torch.Tensor,
    pos_right: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Manhattan-distance hard-negative mining, per OpenEA's ``get_neg``.

    For each seed pair's left/right anchor entity, the ``k`` nearest
    (L1/cityblock distance) *other* entities in the current embedding
    space become that side's negatives -- unlike GCN-Align's uniform
    random sampling. Uses `torch.cdist` (GPU-friendly) rather than
    `scipy.spatial.distance.cdist` -- with a real seed-pair count in the
    thousands and an entity count in the tens of thousands, computing this
    on CPU every refresh would be a genuine bottleneck (same "vectorize/
    accelerate before trusting a smaller-scale precedent" lesson as
    `_bootea_training.encode_triples`'s profiling note).

    **Deliberate deviation**: OpenEA's own ``get_neg`` never excludes the
    anchor entity itself from its own candidate pool -- since the anchor's
    distance to itself is always the global minimum (``0``), it's very
    often ranked as its own "hardest negative," which is meaningless as a
    training signal (the corresponding loss term becomes ``relu(margin +
    pos_dist)``, effectively a constant that can never be reduced by
    training). This function excludes each anchor's own id from its
    ranking before taking the top ``k`` -- a deliberate fix, not
    reproduced as OpenEA's own apparent oversight would have it.

    Args:
        embeddings: ``(num_entities, dim)`` current entity embeddings.
        pos_left: ``(t,)`` ids, seed pairs' left/source side.
        pos_right: ``(t,)`` ids, seed pairs' right/target side.
        k: Negatives mined per pair per side.

    Returns:
        ``(neg_left, neg_right, neg2_left, neg2_right)``, each ``(t * k,)``
        -- pass directly to
        [margin_ranking_loss_l1][linkingtk.algorithms.ea._ea_losses.margin_ranking_loss_l1].
        ``neg_left``/``neg2_right`` are each anchor's own id repeated
        ``k`` times; ``neg_right``/``neg2_left`` are the hard-mined
        neighbor ids.
    """
    import torch

    def _hard_neighbors(anchor_ids: torch.Tensor) -> torch.Tensor:
        anchor_vecs = embeddings[anchor_ids]
        distances = torch.cdist(anchor_vecs, embeddings, p=1)
        distances.scatter_(1, anchor_ids.unsqueeze(1), float("inf"))
        ranked = torch.topk(distances, k, dim=1, largest=False).indices
        return ranked.reshape(-1)

    neg_right = _hard_neighbors(pos_left)
    neg2_left = _hard_neighbors(pos_right)
    neg_left = pos_left.repeat_interleave(k)
    neg2_right = pos_right.repeat_interleave(k)
    return neg_left, neg_right, neg2_left, neg2_right
