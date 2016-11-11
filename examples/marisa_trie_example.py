import marisa_trie

trie = marisa_trie.Trie(["a", "b", "c"])


print(trie.key_id("a"))
print(trie.key_id("b"))
print(trie.key_id("c"))