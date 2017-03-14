from tensorx.layers import Input
from tensorx.models.nrp import NRPRegression, NRPCBow
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
k = 1000
s = 4
h_dim = 10
ri_gen = RIGen(active=s, dim=k)
perm_generator = PermutationGenerator(dim=k)
perm = perm_generator.matrix()

# model definition
#reg_labels = Input(n_units=k, name="ri_output")
#reg_model = NRPRegression(k_dim=k, h_dim=h_dim)
#reg_loss = reg_model.get_loss(reg_labels)

prob_label = Input(n_units=k * 2, name="ri_classes")
prob_model = NRPCBow(k_dim=k, h_dim=h_dim)
prob_loss = prob_model.get_loss(prob_label)

optimizer = tf.train.AdagradOptimizer(learning_rate=1)
train_step = optimizer.minimize(prob_loss)
init = tf.global_variables_initializer()

toy_corpus = [
    "A 1",
    "B 1",
    "B 2",
    "C 2"
]

sentences = [s.split() for s in toy_corpus]

vocab = set()
for sentence in sentences:
    vocab.update(sentence)

# create toy index

index = TrieSignIndex(ri_gen, vocab, pregen_indexes=True)
print("vocabulary size: ", len(index))

window_size = 1

ss = tf.Session()
ss.run(init)
embeddings = ss.run(prob_model.h.weights)
print("embeddings [{min} {max}]".format(min=np.min(embeddings), max=np.max(embeddings)))

i = 1


x_samples = []
y_samples = []

# repeat for n epochs
for epoch in repeat(sentences, 10):
    #print(i)
    for sentence in epoch:
        windows = sliding_windows(sentence, window_size)

        for window in windows:
            word_t = window.target
            ctx_ri = to_bow(window, index, include_target=False, normalise=True)

            ri_t = index.get_ri(word_t).to_dist_vector()

            x_samples.append(ctx_ri)
            y_samples.append(ri_t)

        #print(len(x_samples))
        #print(len(y_samples))

        current_loss = ss.run(tf.reduce_mean(prob_loss), {
                prob_model.input(): x_samples,
                prob_label(): y_samples
        })

        print(current_loss)

        for i in range(10):
            ss.run(train_step, {
                prob_model.input(): x_samples,
                prob_label(): y_samples
            })

        x_samples = []
        y_samples = []


    print("=============================")
    i += 1

# cosine similarity == dot product if we normalize the embeddings
embeddings = ss.run(tf.nn.l2_normalize(prob_model.h.weights, dim=1))
print("embeddings [{min} {max}]".format(min=np.min(embeddings), max=np.max(embeddings)))

print(embeddings.shape)
A = np.matmul(index.get_ri("A").to_vector(), embeddings)
B = np.matmul(index.get_ri("B").to_vector(), embeddings)
C = np.matmul(index.get_ri("C").to_vector(), embeddings)


C2 = np.matmul(index.get_ri("2").to_vector(), embeddings)
C1 = np.matmul(index.get_ri("1").to_vector(), embeddings)


print("A---A ", cosine(A, A))
print("A---B ", cosine(A, B))
print("A---C ", cosine(A, C))

print("\n")
print("C1---A", cosine(C1, A))
print("C1---B", cosine(C1, B))
print("C1---C", cosine(C1, C))
print("C1---C2", cosine(C1, C2))

ss.close()



# test model training
