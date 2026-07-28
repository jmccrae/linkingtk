from linkingtk.blocking.trie import PatriciaTrie, edit_distance


def test_edit_distance_known_values() -> None:
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3
    assert edit_distance("abc", "abc") == 0
    assert edit_distance("abc", "") == 3


def test_edit_distance_is_symmetric() -> None:
    assert edit_distance("flaw", "lawn") == edit_distance("lawn", "flaw")


def test_nearest_finds_exact_match_first() -> None:
    trie: PatriciaTrie[str] = PatriciaTrie()
    for word in ["color", "colour", "colonel", "dog", "cat", "category"]:
        trie.insert(word, word)

    results = trie.nearest("color", 3, queue_max=100)

    assert results[0] == ("color", 0.0)
    assert [value for value, _ in results] == ["color", "colour", "colonel"]
    assert all(results[i][1] <= results[i + 1][1] for i in range(len(results) - 1))


def test_nearest_respects_max_matches() -> None:
    trie: PatriciaTrie[str] = PatriciaTrie()
    for word in ["aaa", "aab", "aac", "aad", "aae"]:
        trie.insert(word, word)

    assert len(trie.nearest("aaa", 2, queue_max=100)) == 2


def test_nearest_supports_duplicate_keys() -> None:
    trie: PatriciaTrie[str] = PatriciaTrie()
    trie.insert("cat", "cat-1")
    trie.insert("cat", "cat-2")

    results = trie.nearest("cat", 5, queue_max=100)

    assert {value for value, _ in results} == {"cat-1", "cat-2"}
    assert all(score == 0.0 for _, score in results)


def test_nearest_on_empty_trie_returns_empty() -> None:
    trie: PatriciaTrie[str] = PatriciaTrie()
    assert trie.nearest("anything", 3, queue_max=100) == []
