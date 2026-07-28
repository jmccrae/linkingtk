"""A Patricia (radix) trie with approximate nearest-neighbor lookup.

Ported from Naisc's ``ApproximateStringMatching.PatriciaTrie``:
https://github.com/insight-centre/naisc/blob/master/naisc-core/src/main/java/org/insightcentre/uld/naisc/blocking/ApproximateStringMatching.java

The trie stores strings compressed along shared prefixes so that a
best-first search, guided by a lower bound on normalized Levenshtein
distance, can find the nearest keys to a query without scoring every
entry in the trie.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


def edit_distance(s: str, t: str) -> int:
    """Levenshtein edit distance between two strings."""
    if len(s) > len(t):
        s, t = t, s
    previous = list(range(len(s) + 1))
    for j in range(1, len(t) + 1):
        current = [previous[0] + 1] + [0] * len(s)
        for i in range(1, len(s) + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            current[i] = min(previous[i] + 1, current[i - 1] + 1, previous[i - 1] + cost)
        previous = current
    return previous[len(s)]


def _edit_distance_lower_bound(left: str, right: str) -> int:
    """Lower bound of the edit distance between ``left`` and any string starting with ``right``."""
    previous = list(range(len(left) + 1))
    for j in range(1, len(right) + 1):
        current = [previous[0] + 1] + [0] * len(left)
        for i in range(1, len(left) + 1):
            cost = 0 if left[i - 1] == right[j - 1] else 1
            current[i] = min(previous[i] + 1, current[i - 1] + 1, previous[i - 1] + cost)
        previous = current
    return min(max(len(left), len(right)), min(previous))


def _normalized_lower_bound(query: str, prefix: str) -> float:
    if len(query) <= len(prefix):
        return edit_distance(query, prefix) / (len(query) + len(prefix))
    diff = len(query) - len(prefix)
    return _edit_distance_lower_bound(query, prefix) / (len(query) + len(prefix) + diff)


def _common_prefix_length(s: str, t: str) -> int:
    limit = min(len(s), len(t))
    for i in range(limit):
        if s[i] != t[i]:
            return i
    return limit


@dataclass
class _TrieNode(Generic[T]):
    children: list[_TrieLink[T]] = field(default_factory=list)
    values: list[T] = field(default_factory=list)


@dataclass
class _TrieLink(Generic[T]):
    link: str
    node: _TrieNode[T]


class PatriciaTrie(Generic[T]):
    """Radix trie mapping strings to (possibly repeated) values."""

    def __init__(self) -> None:
        self._root: _TrieNode[T] = _TrieNode()

    def insert(self, key: str, value: T) -> None:
        """Associate ``value`` with ``key``. Empty keys are ignored."""
        if key:
            self._insert(key, value, self._root)

    def _insert(self, key: str, value: T, node: _TrieNode[T]) -> None:
        for child in node.children:
            i = _common_prefix_length(key, child.link)
            if 0 < i < len(child.link):
                node.children.remove(child)
                common = key[:i]
                if i < len(key):
                    fork_children = [
                        _TrieLink(key[i:], _TrieNode(values=[value])),
                        _TrieLink(child.link[i:], child.node),
                    ]
                    node.children.append(_TrieLink(common, _TrieNode(children=fork_children)))
                else:
                    fork_children = [_TrieLink(child.link[i:], child.node)]
                    node.children.append(
                        _TrieLink(common, _TrieNode(children=fork_children, values=[value]))
                    )
                return
            if i == len(child.link):
                if i == len(key):
                    child.node.values.append(value)
                else:
                    self._insert(key[i:], value, child.node)
                return
        node.children.append(_TrieLink(key, _TrieNode(values=[value])))

    def nearest(self, query: str, n: int, queue_max: int) -> list[tuple[T, float]]:
        """Find up to ``n`` values whose keys best match ``query``.

        Uses best-first search over the trie, pruned by a lower bound on
        normalized edit distance, so that only a bounded, promising
        portion of the trie is actually scored.

        This is an approximate search, as in the original Naisc
        implementation: the lower bound assumes the shortest possible
        completion of each trie prefix, so a true match reached through a
        much longer key can occasionally be pruned before it is scored.
        For blocking (where the goal is a good candidate set, not a
        provably-exact top-n) this is an acceptable and deliberate
        speed/recall trade-off.

        Args:
            query: The string to search for.
            n: Maximum number of results to return.
            queue_max: Maximum size of the search frontier; bounds
                worst-case search cost at some risk of missing a match.

        Returns:
            Up to ``n`` ``(value, normalized_edit_distance)`` pairs,
            sorted by ascending distance (closest match first).
        """
        if n < 1:
            return []
        counter = itertools.count()
        frontier: list[tuple[float, int, str, _TrieNode[T]]] = []
        for child in self._root.children:
            lb = _normalized_lower_bound(query, child.link)
            heapq.heappush(frontier, (lb, next(counter), child.link, child.node))

        beam: list[tuple[float, int, T]] = []

        def worst() -> float:
            return -beam[0][0] if beam else float("inf")

        while frontier:
            if len(beam) >= n and frontier[0][0] > worst():
                break
            _, _, key, node = heapq.heappop(frontier)
            for child in node.children:
                combined = key + child.link
                child_lb = _normalized_lower_bound(query, combined)
                if len(beam) < n or child_lb < worst():
                    heapq.heappush(frontier, (child_lb, next(counter), combined, child.node))
            if len(frontier) > queue_max * 2:
                # nsmallest's output is already sorted, which trivially satisfies
                # the heap invariant, so no separate heapify is needed. Trimming
                # only once the frontier has grown well past queue_max (rather
                # than as soon as it's exceeded) amortizes this O(log) pass
                # across many iterations instead of paying it almost every time.
                frontier[:] = heapq.nsmallest(queue_max, frontier)

            denom = len(query) + len(key)
            score = edit_distance(query, key) / denom if denom else 0.0
            for value in node.values:
                if len(beam) < n or score < worst():
                    heapq.heappush(beam, (-score, next(counter), value))
                    if len(beam) > n:
                        heapq.heappop(beam)

        results = [(value, -neg_score) for neg_score, _, value in beam]
        results.sort(key=lambda item: item[1])
        return results
