import marisa_trie

s = ["A", "AC", "B"]
trie =  marisa_trie.Trie(list(s))
assert(len(trie) == 3)

keys = trie.items()

ws,ids = zip(*keys)


for w in ws:
    print(w)

for ids in ids:
    print(ids)
