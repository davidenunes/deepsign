import os
import sys
import h5py
from tqdm import tqdm

from experiments.pipe.bnc_pipe import BNCPipe
from deepsign.utils.views import chunk_it
from deepsign.rp.index import TrieSignIndex
from deepsign.rp.ri import Generator

from deepsign.utils.views import sliding_windows

from deepsign.rp.encode import to_bow
from deepsign.nlp.utils import subsamplig_prob_cut as ss_prob

import numpy as np
import tensorflow as tf
from tensorx.models.nrp import NRP
from tensorx.layers import Input

# ======================================================================================
# util fn
# ======================================================================================


# ======================================================================================
# Parameters
# ======================================================================================
# random indexing
k = 2000  # random index dim
s = 10  # num active indexes

# context windows
window_size = 2  # sliding window size
subsampling = True
freq_cut = pow(10, -4)

# neural net
h_dim = 600  # dimension for hidden layer
batch_size = 50

# output
home = os.getenv("HOME")
result_dir = home + "/data/results/"
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
pos_labels = Input(n_units=k, name="yp")
neg_labels = Input(n_units=k, name="yn")

model = NRP(k_dim=k, h_dim=h_dim)
loss = model.get_loss(pos_labels, neg_labels)

# turn off norm regularisation
#loss = loss + model.embedding_regularisation(weight=0.001)
#loss = loss + model.output_regularisation(weight=0.001)

perplexity = model.get_perplexity(pos_labels, neg_labels)

learning_rate = 0.05
optimizer = tf.train.AdagradOptimizer(learning_rate)
train_step = optimizer.minimize(loss)
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
            yp = y.copy()
            yp[yp < 0] = 0
            yn = y.copy()
            yn[yn > 0] = 0
            yn = np.abs(yn)

            # current perplexity
            if i % 10000 == 0:
                current_perplexity = tf_session.run(perplexity, {
                    model.input(): x,
                    pos_labels(): yp,
                    neg_labels(): yn
                })
                print("\nbatch shape: ", x.shape)
                print("perplexity:", current_perplexity)

            # train step
            for e in range(1):
                tf_session.run(train_step, {
                    model.input(): x,
                    pos_labels(): yp,
                    neg_labels(): yn
                })

            x_samples = []
            c_samples = []

    # train last batch
    if len(x_samples) > 0:
        # train step
        for e in range(1):
            tf_session.run(train_step, {
                model.input(): x,
                pos_labels(): yp,
                neg_labels(): yn
            })

        x_samples = []
        c_samples = []

    corpus_hdf5.close()
    vocab_hdf5.close()

    print("saving model")
    # save random indexes and model
    sign_index.save(index_file)
    model.save(tf_session, filename=model_file)

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
