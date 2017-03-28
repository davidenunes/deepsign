from tensorx.layers import Input
from tensorx.models.nrp import NRPRegression, NRP
from deepsign.rp.ri import Generator as RIGen
import numpy as np
from deepsign.utils.views import sliding_windows
from deepsign.rp.encode import to_bow, to_bow_dir, to_bow_order
from deepsign.rp.index import TrieSignIndex
from itertools import repeat
from deepsign.utils.measure import cosine
from deepsign.rp.permutations import PermutationGenerator
from tqdm import tqdm

import tensorflow as tf

# random index dimension
k = 1000
s = 10
h_dim = 300
ri_gen = RIGen(active=s, dim=k)
perm_generator = PermutationGenerator(dim=k)
perm = perm_generator.matrix()

# model definition
#reg_labels = Input(n_units=k, name="ri_output")
#reg_model = NRPRegression(k_dim=k, h_dim=h_dim)
#reg_loss = reg_model.get_loss(reg_labels)

prob_label = Input(n_units=k * 2, name="ri_classes")
#prob_model = NRPCBow(k_dim=k, h_dim=h_dim, h_init=tf.zeros)
prob_model = NRP(k_dim=k, h_dim=h_dim,h_init=tf.zeros)
#prob_model = NRP(k_dim=k, h_dim=h_dim)
#prob_loss = prob_model.get_loss(prob_label)
prob_loss = prob_model.get_softmax_loss(prob_label)

# zero init seems to work better for embeddings
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.2)
train_step = optimizer.minimize(prob_loss)
init = tf.global_variables_initializer()

toy_corpus = [
    "1 A 2",
    "1 B 2",
    "1 C 2",
    "1 D 3",
    "1 E 3",
    "4 F 5"
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
#ep = 250 with gradient descend 0.2
#ep = 200 minimum for gradient descend
#ep = 250 with adagrad also works 0.2
#ep = 50 minimum for adagrad 0.2
#ep = 10000 with adadelta
#ep = 3000 minimum with adadelta
# rmsprop too slow
# ep = 270 min for adam 0.001 Adam works well for online learning??
ep = 300

for epoch in tqdm(repeat(sentences, ep),total=ep):
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

        #if i % 1000 == 0:
        #    current_loss = ss.run(tf.reduce_mean(prob_loss), {
        #            prob_model.input(): x_samples,
        #            prob_label(): y_samples
        #    })

            #print(current_loss)

        for i in range(1):
            ss.run(train_step, {
                prob_model.input(): x_samples,
                prob_label(): y_samples
            })

        x_samples = []
        y_samples = []


    #print("=============================")
    i += 1

# A should be similar to  B

# cosine similarity == dot product if we normalize the embeddings
#embeddings = ss.run(tf.nn.l2_normalize(prob_model.h.weights, dim=1))
embeddings = ss.run(prob_model.h.weights)
print("embeddings [{min} {max}]".format(min=np.min(embeddings), max=np.max(embeddings)))

print(embeddings.shape)
A = np.matmul(index.get_ri("A").to_vector(), embeddings)
B = np.matmul(index.get_ri("B").to_vector(), embeddings)
C = np.matmul(index.get_ri("C").to_vector(), embeddings)
D = np.matmul(index.get_ri("D").to_vector(), embeddings)
E = np.matmul(index.get_ri("E").to_vector(), embeddings)
F = np.matmul(index.get_ri("F").to_vector(), embeddings)



C1 = np.matmul(index.get_ri("1").to_vector(), embeddings)
C2 = np.matmul(index.get_ri("2").to_vector(), embeddings)
C5 = np.matmul(index.get_ri("5").to_vector(), embeddings)


print("A---A ", cosine(A, A))
print("A---B ", cosine(A, B))
print("A---C ", cosine(A, C))
print("A---D ", cosine(A, D))
print("D---E ", cosine(D, E))
print("A---F ", cosine(A, F))
print("B---F ", cosine(B, F))

print("\n")
print("1---A", cosine(C1, A))
print("1---B", cosine(C1, B))
print("1---C", cosine(C1, C))
print("1---2", cosine(C1, C2))
print("1---5", cosine(C1, C5))

ss.close()



# test model training
