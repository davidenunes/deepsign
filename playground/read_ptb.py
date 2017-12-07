import os
from deepsign.io.corpora.ptb import PTBReader
from collections import Counter
from deepsign.utils.views import ngram_windows

home = os.getenv("HOME")
ptb_dir = home+"/data/gold_standards/ptb"

reader = PTBReader(ptb_dir)

training_set = reader.training_set()

vocab = Counter()

for sentence in training_set:
    words = sentence
    ngrams = ngram_windows(words,4)
    for ngram in ngrams:
        print(ngram)
    vocab.update(words)

print("total words:",sum(vocab.values()))
print("total unique words:",len(vocab.keys()))
print("100 most common:")
for w in vocab.most_common(100):
    print(w)


