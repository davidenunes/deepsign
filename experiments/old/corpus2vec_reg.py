import os
import sys
from functools import partial

import h5py
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from deepsign.io.corpora.pipe import BNCPipe
from deepsign.nlp.utils import subsamplig_prob_cut as ss_prob
from deepsign.rp.encode import to_bow
from deepsign.rp.index import TrieSignIndex
from deepsign.rp.ri import Generator
from deepsign.utils.views import chunk_it
from deepsign.utils.views import sliding_windows
from tensorx_old.layers import Input
from tensorx_old.models.nrp2 import NRPRegression

# ======================================================================================
# util fn
# ======================================================================================


# ======================================================================================
# Parameters
# ======================================================================================
# random indexing
k = 2000  # random index dim
s = 10    # num active indexes

# context windows
window_size = 2  # sliding window size
subsampling = True
freq_cut = pow(10, -4)

# neural net
h_dim = 600  # dimension for hidden layer
batch_size = 10
learning_rate = 0.2


# output
home = os.getenv("HOME")
result_dir = home + "/data/results/regression/"
index_file = result_dir + "index.hdf5"
model_file = result_dir + "model_bnc"


# ======================================================================================
# Load Corpus
# ======================================================================================
data_dir = home + "/data/datasets/"
corpus_file = data_dir + "bnc_full.hdf5"

corpus_hdf5 = h5py.File(corpus_file, 'r')
corpus_dataset = corpus_hdf5["sentences"]
# iterates over lines but loads them as chunks
# n_rows = 100000
# sentences = chunk_it(corpus_dataset,n_rows=n_rows, chunk_size=40000)
n_rows = len(corpus_dataset)
sentences = chunk_it(corpus_dataset, chunk_size=100000)

pipeline = BNCPipe(datagen=sentences)
# ======================================================================================
# Load Vocabulary
# ======================================================================================
vocab_file = data_dir + "bnc_vocab_spacy.hdf5"
vocab_hdf5 = h5py.File(vocab_file, 'r')

ri_gen = Generator(dim=k, active=s)
print("Loading Vocabulary...")
sign_index = TrieSignIndex(ri_gen, list(vocab_hdf5["vocabulary"][:]), pregen_indexes=False)

if subsampling:
    freq = TrieSignIndex.map_frequencies(list(vocab_hdf5["vocabulary"][:]),
                                         list(vocab_hdf5["frequencies"][:]),
                                         sign_index)

    total_freq = np.sum(vocab_hdf5["frequencies"])

print("done")

# ======================================================================================
# Neural Random Projections Model
# ======================================================================================
labels = Input(n_units=k, name="labels")


# embedding init
h_init = partial(tf.random_uniform,minval=-1,maxval=1)

model = NRPRegression(k_dim=k, h_dim=h_dim, h_init=h_init)
loss = model.get_loss(labels)

# turn off norm regularisation
#loss = loss + model.embedding_regularisation(weight=0.001)
#loss = loss + model.output_regularisation(weight=0.001)

#perplexity = model.get_perplexity(pos_labels, neg_labels)


optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)
train_step = optimizer.minimize(loss)

#global_step = tf.Variable(0, trainable=False)
#decaying_learning_rate = tf.train.exponential_decay(learning_rate, global_step,100000, 0.96, staircase=True)
#train_step = (
#    tf.train.GradientDescentOptimizer(decaying_learning_rate)
#    .minimize(loss, global_step=global_step)
#)

var_init = tf.global_variables_initializer()

tf_session = tf.Session()

# ======================================================================================
# Process Corpus
# ======================================================================================
try:
    # init model variables
    tf_session.run(var_init)


    def keep_token(token):
        fw = freq[sign_index.get_id(token)]
        p = ss_prob(fw, total_freq)
        if np.random.rand() < p:
            return False
        return True


    if subsampling:
        windows_stream = (sliding_windows(list(filter(keep_token, tokens)), window_size) for tokens in pipeline)
    else:
        windows_stream = (sliding_windows(tokens, window_size) for tokens in pipeline)

    i = 0
    x_samples = []
    c_samples = []

    for windows in tqdm(windows_stream, total=n_rows):
        if len(windows) > 0:
            # list of (target,ctx)
            for window in windows:
                target = sign_index.get_ri(window.target).to_vector()
                ctx = to_bow(window, sign_index, include_target=False, normalise=True)

                x_samples.append(target)
                c_samples.append(ctx)

        i += 1

        # batch size in number of sentences
        if i % batch_size == 0:
            # feed data to the model
            x = np.asmatrix(x_samples)
            y = np.asmatrix(c_samples)


            # current perplexity
            if i % 5000 == 0:
                current_perplexity = tf_session.run(loss, {
                    model.input(): x,
                    labels(): y
                })
                print("\nbatch shape: ", x.shape)
                print("loss:", current_perplexity)

            # train step
            for e in range(1):
                tf_session.run(train_step, {
                    model.input(): x,
                    labels(): y
                })

            x_samples = []
            c_samples = []

    # train last batch
    if len(x_samples) > 0:
        # train step
        for e in range(1):
            tf_session.run(train_step, {
                model.input(): x,
                labels(): y
            })

        x_samples = []
        c_samples = []

    corpus_hdf5.close()
    vocab_hdf5.close()

    print("saving model")
    # save random indexes and model
    sign_index.save(index_file)
    model.save(tf_session, model_filename=model_file)

    print("done")

    tf_session.close()

# ======================================================================================
# Process Interrupted
# ======================================================================================
except (KeyboardInterrupt, SystemExit):
    # TODO store the model current state
    # and the state of the corpus iteration
    print("\nProcess interrupted, closing corpus and saving progress...", file=sys.stderr)
    corpus_hdf5.close()
    vocab_hdf5.close()
    tf_session.close()
else:
    # save the model
    pass
