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

from itertools import repeat

import argparse

# ======================================================================================
# Argument parse configuration
# ======================================================================================
parser = argparse.ArgumentParser(description="Neural Random Projections arguments")
parser.add_argument('-ri_k', dest="ri_k", type=int, default=1000)
parser.add_argument('-ri_s', dest="ri_s", type=int, default=10)
parser.add_argument('-h_dim', dest="h_dim", type=int, default=100)
parser.add_argument('-epochs', dest="epochs", type=int, default=10)
parser.add_argument('-window_size', dest="window_size", type=int, default=2)
parser.add_argument('-subsampling', dest="subsampling", type=bool, default=True)
parser.add_argument('-lemmas', dest="lemmas", type=bool, default=False)
parser.add_argument('-out_dir', dest="out_dir", type=str, default="/data/results/")
parser.add_argument('-freq_cut', dest="freq_cut", type=float, default=pow(10, -5))
parser.add_argument('-learning_rate', dest="learning_rate", type=float, default=0.01)
parser.add_argument('-batch_size', dest="batch_size", type=int, default=10)
parser.add_argument('-n_sentences', dest="n_sentences", type=int, default=5000)
args = parser.parse_args()

print(args)
# ======================================================================================
# Parameters
# ======================================================================================
# random indexing
k = args.ri_k
s = args.ri_s

# context windows
window_size = args.window_size
subsampling = args.subsampling
freq_cut = args.freq_cut

# neural net
h_dim = args.h_dim
# training params
batch_size = args.batch_size

# files
home = os.getenv("HOME")
model_suffix = "{k}_{s}_h{h}".format(k=k, s=s, h=h_dim)
result_dir = home + args.out_dir
index_file = result_dir + "index_" + model_suffix + ".hdf5"
model_file = result_dir + "model_" + model_suffix

print("training: ", index_file)

# ======================================================================================
# Load Corpus
# ======================================================================================
data_dir = home + "/data/datasets/"
corpus_file = data_dir + "bnc.hdf5"

corpus_hdf5 = h5py.File(corpus_file, 'r')
corpus_dataset = corpus_hdf5["sentences"]
# iterates over lines but loads them as chunks
if args.n_sentences == -1:
    n_rows = len(corpus_dataset)
else:
    n_rows = args.n_sentences

# sentences = chunk_it(corpus_dataset,n_rows=n_rows, chunk_size=40000)
# n_rows = len(corpus_dataset)




# ======================================================================================
# Load Vocabulary
# ======================================================================================
if args.lemmas:
    vocab_file = data_dir + "bnc_vocab_lemma.hdf5"
else:
    vocab_file = data_dir + "bnc_vocab.hdf5"

vocab_hdf5 = h5py.File(vocab_file, 'r')

ri_gen = Generator(dim=k, active=s)
print("Loading Vocabulary...")
index = TrieSignIndex(ri_gen, list(vocab_hdf5["vocabulary"][:]), pregen_indexes=False)

if subsampling:
    freq = TrieSignIndex.map_frequencies(list(vocab_hdf5["vocabulary"][:]),
                                         list(vocab_hdf5["frequencies"][:]),
                                         index)

    total_freq = np.sum(vocab_hdf5["frequencies"])

print("done")

# ======================================================================================
# Neural Random Projections Model
# ======================================================================================



# embedding init
model = NRP(k_dim=k, h_dim=h_dim)
#model = NRP(k_dim=k, h_dim=h_dim, h_init=tf.zeros)
labels = Input(n_units=k * 2, name="ri_classes")
loss = model.get_loss(labels)
#loss = model.get_softmax_loss(labels)

learning_rate = args.learning_rate
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
train_step = optimizer.minimize(loss)

var_init = tf.global_variables_initializer()
sess = tf.Session()
# init model variables
sess.run(var_init)


# ======================================================================================
# Process Corpus
# ======================================================================================
def keep_token(token):
    fw = freq[index.get_id(token)]
    p = ss_prob(fw, total_freq)
    if np.random.rand() < p:
        return False
    return True


def get_window_stream(pipeline):


    if subsampling:
        windows_stream = (sliding_windows(list(filter(keep_token, tokens)), window_size) for tokens in pipeline)
    else:
        windows_stream = (sliding_windows(tokens, window_size) for tokens in pipeline)

    return windows_stream


try:
    sentences = chunk_it(corpus_dataset, n_rows=n_rows, chunk_size=100000)
    pipeline = BNCPipe(datagen=sentences, lemmas=args.lemmas)

    for epoch in range(args.epochs):
        print("epoch ", epoch + 1)
        i = 0
        x_samples = []
        y_samples = []

        # restart sentence iterator
        sentences = chunk_it(corpus_dataset, n_rows=n_rows, chunk_size=10000)
        pipeline.reaload(sentences)
        window_stream = get_window_stream(pipeline)

        for windows in tqdm(window_stream, total=n_rows):
            if len(windows) > 0:
                # list of (target,ctx)
                for window in windows:
                    word_t = window.target
                    ctx_ri = to_bow(window, index, include_target=False, normalise=True)
                    target_vector = index.get_ri(word_t).to_dist_vector()

                    x_samples.append(ctx_ri)
                    y_samples.append(target_vector)

            i += 1

            # batch size in number of sentences
            if i % batch_size == 0 and len(y_samples) > 0:
                # print current loss
                if i % 1000 == 0:
                    current_loss = sess.run(loss, {
                        model.input(): x_samples,
                        labels(): y_samples,
                    })
                    print("loss: ", current_loss)

                # train step
                sess.run(train_step, {
                    model.input(): x_samples,
                    labels(): y_samples,
                })

                x_samples = []
                y_samples = []

        # run train on last batch
        if len(x_samples) > 0:
            sess.run(train_step, {
                model.input(): x_samples,
                labels(): y_samples,
            })

            x_samples = []
            y_samples = []

    corpus_hdf5.close()
    vocab_hdf5.close()

    print("saving model")
    # save random indexes and model
    index.save(index_file)
    model.save(sess, model_filename=model_file, embeddings_name=model_suffix)

    print("done")

    sess.close()

# ======================================================================================
# Process Interrupted
# ======================================================================================
except (KeyboardInterrupt, SystemExit):
    # TODO save current model progress
    print("\nProcess interrupted, closing corpus and saving progress...", file=sys.stderr)
    corpus_hdf5.close()
    vocab_hdf5.close()
    sess.close()
else:
    # save the model
    pass
