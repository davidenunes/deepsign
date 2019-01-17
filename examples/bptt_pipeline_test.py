from tensorx.data import itertools as itx
import numpy as np
from deepsign.data.pipelines import to_parallel_seq

n = 100
bsz = 5
seq = 10
data_ids = np.arange(n, dtype=np.int32)
data_str = data_ids.astype('U')
vocab = dict(zip(data_str, data_ids))

data_it = to_parallel_seq(corpus_fn=lambda: iter(data_str),
                          vocab=vocab,
                          epochs=1,
                          seq_len=10,
                          enum_seq=False,
                          batch_size=bsz,
                          enum_epoch=True)

data = next(data_it)

# print(i)
print(data)
