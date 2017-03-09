from tensorx.layers import Input
from tensorx.models.nrp import NRPRegression, NRPSkipReg
from deepsign.rp.ri import Generator as RIGen
import numpy as np
from deepsign.utils.views import sliding_windows
from deepsign.rp.encode import to_bow, to_bow_dir, to_bow_order
from deepsign.rp.index import TrieSignIndex
from itertools import repeat
from deepsign.utils.measure import cosine
from deepsign.rp.permutations import PermutationGenerator

import tensorflow as tf

# random index dimension
k = 2000
s = 10
h_dim = 600
ri_gen = RIGen(active=s, dim=k)
perm_generator = PermutationGenerator(dim=k)
perm = perm_generator.matrix()



labels = Input(n_units=k, name="ri")

model = NRPRegression(k_dim=k,h_dim=h_dim)
loss = model.get_loss(labels)


#modelskip = NRPSkipReg(k_dim=k,h_dim=h_dim,window_reach=1)


optimizer = tf.train.AdadeltaOptimizer(learning_rate=1)
train_step = optimizer.minimize(loss)
init = tf.global_variables_initializer()

toy_corpus = [
    "A 3",
    "A 1",
    "B 1",
    "B 3",
    "B 2",
    "C 3",
    "C 2",
    "D 4",
    "D 4"
]

sentences = [s.split() for s in toy_corpus]

vocab = set()
for sentence in sentences:
    vocab.update(sentence)

# create toy index

index = TrieSignIndex(ri_gen,vocab,pregen_indexes=True)
print("vocabulary size: ",len(index))

window_size = 1

ss = tf.Session()
ss.run(init)
embeddings = ss.run(model.h.weights)
print("embeddings [{min} {max}]".format(min=np.min(embeddings),max=np.max(embeddings)))

i = 1
# repeat for n epochs
for epoch in repeat(sentences,100):
    for sentence in epoch:
        windows = sliding_windows(sentence,window_size)

        for window in windows:
            word_t = window.target
            ctx_ri = to_bow(window,index,include_target=False,normalise=True)
            #ctx_ri = to_bow_dir(window,index,perm_matrix=perm)

            ri_t = index.get_ri(word_t)

            current_loss = ss.run(loss, {
                model.input(): [ri_t.to_vector()],
                labels(): [ctx_ri]
            })
            #print("current loss: ", current_loss)

            ss.run(train_step, {
                model.input(): [ri_t.to_vector()],
                labels(): [ctx_ri]
            })
    i+=1

# cosine similarity == dot product if we normalize the embeddings
embeddings = ss.run(tf.nn.l2_normalize(model.h.weights,dim=1))
print("embeddings [{min} {max}]".format(min=np.min(embeddings),max=np.max(embeddings)))

print(embeddings.shape)
A = np.matmul(index.get_ri("A").to_vector(),embeddings)
B = np.matmul(index.get_ri("B").to_vector(),embeddings)
C = np.matmul(index.get_ri("C").to_vector(),embeddings)
D = np.matmul(index.get_ri("D").to_vector(),embeddings)
C3 = np.matmul(index.get_ri("3").to_vector(),embeddings)
C2 = np.matmul(index.get_ri("2").to_vector(),embeddings)
C1 = np.matmul(index.get_ri("1").to_vector(),embeddings)
C4 = np.matmul(index.get_ri("4").to_vector(),embeddings)


print("A---A ", cosine(A,A))
print("A---B ", cosine(A,B))
print("A---C ", cosine(A,C))
print("A---D ", cosine(A,D))
print("B---D ", cosine(B,D))
print("C---D ", cosine(C,D))
print("\n")
print("C3---A", cosine(C3,A))
print("C3---B", cosine(C3,B))
print("C3---C", cosine(C3,C))
print("C3---D", cosine(C3,D))

print("\n")
print("C1---A", cosine(C1,A))
print("C1---B", cosine(C1,B))
print("C1---C", cosine(C1,C))
print("C1---D", cosine(C1,D))
print("C1---C2", cosine(C1,C2))







ss.close()



# test model training
