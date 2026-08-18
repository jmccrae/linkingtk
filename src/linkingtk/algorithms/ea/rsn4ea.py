"""RSN4EA knowledge-graph-embedding linker for Entity Alignment.

Guo, L., Sun, Z., & Hu, W. (2019). Learning to Exploit Long-term
Relational Dependencies in Knowledge Graphs. ICML 2019.
http://proceedings.mlr.press/v97/guo19c/guo19c.pdf

Structurally the most different method in this family: instead of a
triple-local translational embedding (TransE-style, like every other
linker in ``linkingtk.algorithms.ea``), RSN4EA trains a **Recurrent
Skipping Network** -- an LSTM run over sampled random-walk *paths* through
the combined graph, with a next-token prediction loss (entity/relation
language-modeling, not a distance loss), and a residual "skip" connection
letting each relation-step's prediction shortcut directly from the
preceding entity's embedding. Cross-KG alignment comes from a
data-construction trick, not a mapping matrix or a shared-id merge: seed
pairs make their two entity ids mutually substitutable throughout the
walkable graph, so a random walk that reaches one can carry on through the
other KG's own edges from that point.

Like [MTransELinker][linkingtk.algorithms.ea.mtranse.MTransELinker], this
is a faithful reimplementation of RSN4EA's actual training procedure,
ported directly from OpenEA's reference implementation
(https://github.com/nju-websoft/OpenEA/blob/master/src/openea/approaches/rsn4ea.py)
rather than from a from-scratch reading of the paper. Several of OpenEA's
published config values turn out to be dead -- never actually consulted by
`rsn4ea.py`'s own code -- and this port follows what's *actually executed*:

- `ent_l2_norm`/`rel_l2_norm` are `true` in OpenEA's config, but
  `_define_variables` builds embeddings via plain `tf.get_variable(...,
  xavier_initializer)`, never through the `init_embeddings()` utility that
  applies `l2_normalize` -- **unlike every other linker in this family,
  RSN4EA's embeddings are never L2-normalized**, at init or per-forward-pass.
- `"optimizer": "Adagrad"` in the config, but `_define_variables` hardcodes
  `tf.train.AdamOptimizer(options.learning_rate)` directly, bypassing
  `generate_optimizer(..., opt=self.args.optimizer)` -- **Adam is what
  actually runs**, so this port uses Adam unconditionally (not a
  constructor option).
- `"dim": 100` is never read anywhere in `rsn4ea.py` -- embedding size is
  driven entirely by `hidden_size`.
- `add_weight`'s `w_h`/`w_r`/`w_t` alias-tracking columns are computed but
  never consumed by `sample_paths`/`cal_loss` -- only the *structural*
  effect of alias substitution (which triples exist) is load-bearing.
- `add_weight`'s relation-alias variants are always empty (RSN4EA's own
  `rel_mapping` is always an empty table -- no relation-alignment signal
  exists for this task), so only the 4 live head/tail-alias variants are
  ported (see
  [_rsn4ea_training.build_augmented_kb][linkingtk.algorithms.ea._rsn4ea_training.build_augmented_kb]).
- `alignment_module: "mapping"` has zero effect on RSN4EA's own training
  (`run()` fully overrides `BasicModel.run()`, never calling
  `launch_training_1epo`'s mapping-loss branch) -- it only matters at the
  id-space level, which is exactly `build_id_mappings`'s disjoint-per-KG
  shape (matching
  [MTransELinker][linkingtk.algorithms.ea.mtranse.MTransELinker]/
  [SEALinker][linkingtk.algorithms.ea.sea.SEALinker]), used here too.

Ported into two private helper modules: `_rsn4ea_training.py` (plain
numpy -- graph augmentation and path sampling, independently testable
without `torch`) and `_rsn4ea_torch.py` (the model and training-step
functions that build/consume PyTorch tensors).

Deliberate deviations from OpenEA's own code, beyond what those two
modules already document:

- **Full softmax cross-entropy instead of `tf.nn.nce_loss`.** OpenEA's
  NCE loss approximates a full softmax via log-uniform negative sampling
  -- a 2019-TF1-era efficiency hack for large vocabularies. At this
  dataset's real scale (~15K entities, low hundreds of relations doubled),
  OpenEA's own `num_samples` is already capped to `vocab_size // 3` for
  entities -- nearly a third of the whole vocabulary sampled as noise every
  step regardless. A full softmax is exact (no noise-distribution
  correction term needed) and tractable at this vocab size on current GPU
  hardware -- the same "swap a reference-specific efficiency mechanism for
  a modern equivalent once its own justification no longer applies at this
  scale" precedent as
  [BootEALinker][linkingtk.algorithms.ea.bootea.BootEALinker]'s
  `scipy.optimize.linear_sum_assignment` substitution for
  `graph_tool`/`igraph`. `num_samples` is therefore not exposed as a
  constructor parameter.
- **No disk-cached path table.** OpenEA caches sampled paths to
  `{data_path}/paths_{alpha}_{beta}` and reloads if present. This repo's
  `fit()` takes in-memory data with no persistent working directory, so
  paths are always freshly sampled each call.
- **`max_paths` cap.** `repeat_times` walks per (already-expanded) row of
  the augmented KB can produce a large path table at real dataset scale.
  An optional `max_paths` cap (random subsample, fixed for the run) is
  exposed as a pragmatic safety valve -- same precedent as
  [IPTransELinker][linkingtk.algorithms.ea.iptranse.IPTransELinker]'s
  `bootstrap_pool_size`. Default `None` (unbounded, matching OpenEA).
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from linkingtk.algorithms.base import DEFAULT_BLOCKING, BaseLinker
from linkingtk.algorithms.ea._device import resolve_device
from linkingtk.algorithms.ea._rsn4ea_torch import build_rsn_model, train_epoch, validation_hits1
from linkingtk.algorithms.ea._rsn4ea_training import build_augmented_kb, sample_paths
from linkingtk.algorithms.matching import DEFAULT_MATCHER, Matcher
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity
from linkingtk.core.result import AlignmentResult
from linkingtk.exceptions import LinkingTKError, OptionalDependencyError
from linkingtk.utils.graph import Graph, build_id_mappings, map_triples_to_ids, to_triples

if TYPE_CHECKING:
    import numpy.typing as npt
    import torch


class RSN4EALinker(BaseLinker):
    """Scores candidate pairs via a Recurrent Skipping Network over sampled walk paths.

    Must be [fit][linkingtk.algorithms.ea.rsn4ea.RSN4EALinker.fit] before
    [link][linkingtk.algorithms.base.BaseLinker.link] can be called. Unlike
    [MTransELinker][linkingtk.algorithms.ea.mtranse.MTransELinker]/
    [SEALinker][linkingtk.algorithms.ea.sea.SEALinker], scoring uses raw
    (unprojected) embeddings on both sides -- RSN4EA never builds a mapping
    matrix; see the module docstring.

    Args:
        hidden_size: Entity/relation embedding dimensionality and LSTM
            hidden size. OpenEA's published EN-FR-15K-V1 config uses
            ``100``.
        num_layers: LSTM depth. OpenEA's published value is ``2``.
        keep_prob: Per-LSTM-layer output dropout keep-probability. OpenEA's
            published value is ``0.6``.
        max_length: Sampled path length -- must be odd and >= 3 (entities
            at even positions, relations at odd positions). OpenEA's
            published value is ``15``.
        alpha: Path-sampling depth-bias weight (down-weights continuations
            already directly reachable from a walk's own start entity).
            OpenEA's published value is ``0.7``.
        beta: Path-sampling cross-KG bias weight (up-weights continuations
            whose tail is a seed-pair entity). OpenEA's published value is
            ``0.7``.
        repeat_times: Walks sampled per row of the augmented knowledge
            base. OpenEA's published value is ``2``.
        max_paths: Optional cap on the number of sampled paths kept for
            training (random subsample, fixed for the run) -- see the
            module docstring. ``None`` (default) keeps every successfully
            sampled path.
        num_epochs: Training epochs. OpenEA's published value is ``30``
            (RSN4EA converges in far fewer epochs than this family's
            TransE-based methods).
        batch_size: Mini-batch size for path training. OpenEA's published
            value is ``512``.
        learning_rate: Adam's learning rate (see the module docstring for
            why Adam, not the config's stated Adagrad). OpenEA's published
            value is ``0.0005``.
        matching: Strategy used to resolve scored candidates into final
            links. Defaults to
            [GreedyMatcher][linkingtk.algorithms.matching.GreedyMatcher].
        device: Torch device to train on, e.g. ``"cpu"`` (default) or
            ``"cuda"``/``"cuda:0"``. Trained embeddings are always stored
            as CPU numpy arrays regardless of this setting.
    """

    def __init__(
        self,
        hidden_size: int = 100,
        num_layers: int = 2,
        keep_prob: float = 0.6,
        max_length: int = 15,
        alpha: float = 0.7,
        beta: float = 0.7,
        repeat_times: int = 2,
        max_paths: int | None = None,
        num_epochs: int = 30,
        batch_size: int = 512,
        learning_rate: float = 0.0005,
        matching: Matcher = DEFAULT_MATCHER,
        device: str = "cpu",
    ) -> None:
        if max_length < 3 or max_length % 2 == 0:
            raise LinkingTKError(f"max_length must be odd and >= 3, got {max_length!r}.")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.keep_prob = keep_prob
        self.max_length = max_length
        self.alpha = alpha
        self.beta = beta
        self.repeat_times = repeat_times
        self.max_paths = max_paths
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.matching = matching
        self.device = device
        self._id_to_vector: dict[str, npt.NDArray[np.floating[Any]]] = {}
        self._fitted = False

    def fit(
        self,
        dataset1: list[Entity],
        dataset2: list[Entity],
        ground_truth: list[tuple[str, str]],
        graph: Graph,
        random_state: int | None = None,
        val_ground_truth: list[tuple[str, str]] | None = None,
        patience: int = 5,
        eval_every: int = 10,
    ) -> RSN4EALinker:
        """Sample walk paths over an alias-expanded graph and train the RSN.

        Args:
            dataset1: Source entities. Unused beyond mirroring sibling
                linkers' signatures -- which entities get embeddings is
                determined entirely by ``graph``.
            dataset2: Target entities. See ``dataset1``.
            ground_truth: List of ``(source_id, target_id)`` known-correct
                pairs. Used to build the alias-substituted graph
                ([build_augmented_kb][linkingtk.algorithms.ea._rsn4ea_training.build_augmented_kb])
                that lets sampled walks cross between KGs -- RSN4EA's only
                alignment-training mechanism, no separate loss channel.
            graph: The combined relational structure of both KGs -- entity
                ids on both sides must already be disjoint, as they are
                from a
                [GraphDatasetLoader][linkingtk.datasets.base.GraphDatasetLoader]'s
                ``load_graphs()``.
            random_state: Seed for reproducible path sampling and training.
                Left unspecified, both are non-deterministic.
            val_ground_truth: Optional held-out pairs used for early
                stopping -- every ``eval_every`` epochs, Hits@1 is checked
                against this set, and training stops after ``patience``
                checks with no improvement. If ``None`` (default), trains
                the full ``num_epochs`` unconditionally.
            patience: Number of non-improving ``eval_every``-spaced checks
                to tolerate before stopping early. Only used if
                ``val_ground_truth`` is given.
            eval_every: How often (in epochs) to check ``val_ground_truth``.
                Only used if ``val_ground_truth`` is given.

        Returns:
            ``self``, for chaining.

        Raises:
            LinkingTKError: If none of ``ground_truth``'s pairs have both
                ids present in ``graph``'s own triples, if no walk path
                could be sampled at all, or if ``device`` is invalid or
                unavailable.
            OptionalDependencyError: If torch isn't installed.
        """
        try:
            import torch
        except ImportError as exc:
            raise OptionalDependencyError("RSN4EALinker", "kge") from exc

        device = resolve_device(self.device)
        if random_state is not None:
            torch.manual_seed(random_state)
            torch.cuda.manual_seed_all(random_state)
        rng = np.random.default_rng(random_state)

        triples = to_triples(graph)
        entity_to_id, relation_to_id = build_id_mappings(triples)
        mapped = map_triples_to_ids(triples, entity_to_id, relation_to_id)

        seed_pairs = [
            (entity_to_id[s], entity_to_id[t])
            for s, t in ground_truth
            if s in entity_to_id and t in entity_to_id
        ]
        if not seed_pairs:
            raise LinkingTKError(
                "None of `ground_truth`'s pairs have both ids present in `graph`'s "
                "own triples; fit() has no seed pairs to build RSN4EA's alias-expanded "
                "graph with."
            )

        augmented_kb, num_relations = build_augmented_kb(
            mapped, len(entity_to_id), len(relation_to_id), seed_pairs
        )
        paths = sample_paths(
            augmented_kb,
            seed_pairs,
            rng,
            self.max_length,
            self.alpha,
            self.beta,
            self.repeat_times,
            self.max_paths,
        )
        if len(paths) == 0:
            raise LinkingTKError(
                "No walk paths of length `max_length` could be sampled from `graph` -- "
                "every entity reachable from a starting triple hit a dead end too soon."
            )

        model: torch.nn.Module = build_rsn_model(
            len(entity_to_id), num_relations, self.hidden_size, self.num_layers, self.keep_prob
        )
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        val_pairs = [
            (s, t) for s, t in (val_ground_truth or []) if s in entity_to_id and t in entity_to_id
        ]
        best_hits1 = -1.0
        epochs_without_improvement = 0
        best_state: dict[str, torch.Tensor] | None = None

        for epoch in range(self.num_epochs):
            train_epoch(model, optimizer, paths, rng, self.batch_size)

            if val_pairs and (epoch + 1) % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    current_embeds = _entity_embedding_weight(model).cpu().numpy()
                hits1 = validation_hits1(current_embeds, entity_to_id, val_pairs)
                if hits1 <= best_hits1:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        break
                else:
                    best_hits1 = hits1
                    epochs_without_improvement = 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        # Unlike this family's other linkers, RSN4EA's validation Hits@1
        # doesn't improve (or plateau) monotonically with more training --
        # diagnosed directly on the real EN-FR-15K-V1 benchmark: it peaks
        # within the first few `eval_every`-spaced checks, then steadily
        # degrades (the next-token training objective is only a proxy for
        # alignment quality, and keeps optimizing well past the point
        # where the two diverge). Restoring the best-seen checkpoint
        # (rather than using whichever epoch training happened to stop or
        # finish on, like every sibling linker here) is what actually
        # fixes this -- not a hyperparameter tweak.
        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            final_embeds = _entity_embedding_weight(model).cpu().numpy()
        self._id_to_vector = {
            entity_id: final_embeds[index] for entity_id, index in entity_to_id.items()
        }
        self._fitted = True
        return self

    def link(
        self,
        dataset1: list[Entity],
        dataset2: list[Entity],
        graph: Graph = None,
        blocking: BlockingStrategy = DEFAULT_BLOCKING,
    ) -> list[AlignmentResult]:
        if not self._fitted:
            raise LinkingTKError("RSN4EALinker.link() called before fit().")

        pairs = blocking.candidate_pairs(dataset1, dataset2)
        target_ids_by_source: dict[str, list[str]] = defaultdict(list)
        for entity1, entity2 in pairs:
            target_ids_by_source[entity1.id].append(entity2.id)

        candidates_by_source: dict[str, list[tuple[str, float]]] = {}
        for source_id, target_ids in target_ids_by_source.items():
            source_vector = self.source_embedding(source_id).reshape(1, -1)
            target_matrix = np.stack([self.target_embedding(target_id) for target_id in target_ids])
            scores = cosine_similarity(source_vector, target_matrix)[0]
            candidates_by_source[source_id] = list(
                zip(target_ids, (float(score) for score in scores), strict=True)
            )

        return self.matching.match(candidates_by_source)

    def source_embedding(self, entity_id: str) -> npt.NDArray[np.floating[Any]]:
        """Vector used to score ``entity_id`` as a scored pair's source side."""
        return self._embedding(entity_id)

    def target_embedding(self, entity_id: str) -> npt.NDArray[np.floating[Any]]:
        """Vector used to score ``entity_id`` as a scored pair's target side."""
        return self._embedding(entity_id)

    def _embedding(self, entity_id: str) -> npt.NDArray[np.floating[Any]]:
        vector = self._id_to_vector.get(entity_id)
        if vector is None:
            raise LinkingTKError(
                f"Entity {entity_id!r} has no trained embedding -- it didn't appear "
                "in fit()'s `graph`."
            )
        return vector


def _entity_embedding_weight(model: torch.nn.Module) -> torch.Tensor:
    """``model.entity_embedding.weight``, typed past a PyTorch stub gap.

    ``torch.nn.Module.__getattr__``'s stub return type is ``Tensor |
    Module`` for any dynamically-registered attribute; mypy can't see
    through [build_rsn_model][linkingtk.algorithms.ea._rsn4ea_torch.build_rsn_model]'s
    ``-> torch.nn.Module`` factory return type to know ``entity_embedding``
    is concretely an ``nn.Embedding``.
    """
    embedding = cast("torch.nn.Embedding", model.entity_embedding)
    return embedding.weight
