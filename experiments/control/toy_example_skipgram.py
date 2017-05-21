from tensorx.models.word2vec import SkipGram
from deepsign.rp.index import TrieSignIndex
from deepsign.rp.ri import Generator
from deepsign.utils.views import sliding_windows
from sklearn.preprocessing import normalize

import gensim

from itertools import repeat
import tensorflow as tf
import numpy as np

toy_corpus = [
    "A 1",
    "B 1",
    "C 1",
    "D 2",
]
sentences = [s.split() for s in toy_corpus]

#gensim
gensim_w2v = gensim.models.Word2Vec(sentences, window=1, min_count=1,iter=10000)
print(gensim_w2v.similarity("A","B"))
print(gensim_w2v.similarity("A","C"))
print(gensim_w2v.similarity("A","1"))
print(gensim_w2v.similarity("A","D"))


vocab = set()
for words in sentences:
    vocab.update(words)

vocab_ids = {}
reverse_ids = {}
for i,w in enumerate(vocab):
    vocab_ids[w] = i
    reverse_ids[i] = w

print("Vocabulary Size:", len(vocab))
# ===============================================
#                      WORD2VEC
# ===============================================
h_dim = 50   # embedding (hidden layer) dimension
skipgram = SkipGram(vocab_size=len(vocab),embedding_dim=h_dim,batch_size=None)
in_placeholder = skipgram.input()
label_placeholder = skipgram.labels
#loss = skipgram.loss()
loss = skipgram.nce_loss()

optimizer = tf.train.AdamOptimizer(0.02)
train_step = optimizer.minimize(loss)
init = tf.global_variables_initializer()

# ===============================================
#                      TRAINING
# ===============================================
print(vocab_ids)
x_samples = []
y_samples = []

losses = []
epochs = 100
with tf.Session() as ss:
    ss.run(init)
    #print(ss.run(normalized_embeddings))
    for i,epoch in enumerate(repeat(sentences,epochs)):
        for sentence in epoch:
            windows = sliding_windows(sentence,window_size=1)
            for window in windows:
                labels = window.left + window.right
                target = window.target
                for label in labels:
                    x_samples.append([vocab_ids[target]])
                    y_samples.append([vocab_ids[label]])


                #print(np.asmatrix(y_samples))
                _, current_loss = ss.run([train_step, loss], {
                    in_placeholder: x_samples,
                    label_placeholder: y_samples
                })

                losses.append(current_loss)

                x_samples = []
                y_samples = []
        mean_loss = np.mean(losses)
        if i % 100 == 0 :
            print("mean loss:", mean_loss)

    norm = tf.sqrt(tf.reduce_sum(tf.square(skipgram.embeddings.weights), 1, keep_dims=True))
    normalized_embeddings = skipgram.embeddings.weights / norm
    print("embeddings:\n", ss.run(normalized_embeddings).shape)
    embeddings = np.zeros([len(vocab), h_dim])
    for w in vocab:
        id = vocab_ids[w]
        embeddings[id,:] = ss.run(skipgram.embeddings(),feed_dict={
            in_placeholder: [[vocab_ids[w]]]
        })
    print("\n\ncodes: \n")
    print(embeddings.shape)

    embeddings /= np.linalg.norm(embeddings, axis=-1)[:, np.newaxis]
    #norm = np.sum(np.sqrt(embeddings),1,keepdims=True)
    #normalised
    #embeddings = embeddings / norm
    #print(embeddings)


    def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
        assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
        plt.figure(figsize=(4, 4))  # in inches
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(label,
                         xy=(x, y),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
        plt.savefig(filename)

    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=200)
        plot_only = len(vocab)
        low_dim_embs = tsne.fit_transform(embeddings[:plot_only, :])
        labels = [reverse_ids[i] for i in range(plot_only)]
        plot_with_labels(low_dim_embs, labels)

    except ImportError:
        print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")



        # use random indexing
#ri_generator = Generator(dim=400,active=20)
#vocab = TrieSignIndex(generator=ri_generator)