import os
from deepsign.data.corpora.wiki103 import WikiText103
from collections import Counter
import numpy as np

home = os.getenv("HOME")
wiki103dir = home + "/data/datasets/wikitext-103"

reader = WikiText103(wiki103dir, mark_eos=True)

data = reader.training_set(n_samples=4)
print(reader.test_file)

vocab = Counter()

print("==== DATA ====")
for i, s in enumerate(data):
    print("{no}.{s}".format(no=i, s=s))
    print(np.array(s,dtype="U"))
print("="*80)
# for ngram in vw.window_it(vw.flatten_it(data), 10):
#    print(ngram)

#   vocab.update(ngram)
print()
print("total words:", sum(vocab.values()))
print("total unique words:", len(vocab.keys()))
print("100 most common:")
# for w in vocab.most_common(100):
#    print(w)
