import os
from deepsign.io.corpora.ptb import PTBReader
from collections import Counter

home = os.getenv("HOME")
ptb_dir = home+"/data/datasets/ptb"

reader = PTBReader(ptb_dir)

training_set = reader.training_set()

vocab = Counter()

for sentence in training_set:
    words = sentence.split()
    vocab.update(words)

print("total words:",sum(vocab.values()))
print("100 most common:")
for w in vocab.most_common(100):
    print(w)


