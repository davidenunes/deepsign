import argparse
import os
import h5py
import marisa_trie
from deepsign.models.nnlm import NNLM

import numpy as np
import tensorflow as tf

from deepsign.data.views import chunk_it
from tensorx.layers import Input

# ======================================================================================
# Argument parse configuration
# -name : model name to be used with result files as a prefix (defaults to nnlm)
# -conf : configuration file path
# -corpus : dataset file path (uses the hdf5 format defined by convert to hdf5 script)
# -output : output dir where results are written, defaults to ~/data/results/
# ======================================================================================
home = os.getenv("HOME")

parser = argparse.ArgumentParser(description="NNLM Baseline Parameters")
parser.add_argument('-conf', dest="conf", type=str)
parser.add_argument('-name', dest="name", type=str, default="nnlm")
parser.add_argument('-corpus', dest="corpus", type=str, default=home + "/data/datasets/ptb/")
parser.add_argument('-output', dest="output", type=str, default=home + "/data/results/")

parser.add_argument('-embed_dim', dest="embed_dim", type=int, default=100)
parser.add_argument('-h_dim', dest="h_dim", type=int, default=100)
parser.add_argument('-epochs', dest="epochs", type=int, default=1)
parser.add_argument('-ngram_size', dest="ngram_size", type=int, default=4)
parser.add_argument('-out_dir', dest="out_dir", type=str, default="/data/results/")
parser.add_argument('-data_dir', dest="data_dir", type=str, default="/data/gold_standards/")
parser.add_argument('-learning_rate', dest="learning_rate", type=float, default=0.01)
parser.add_argument('-batch_size', dest="batch_size", type=int, default=50)
args = parser.parse_args()

out_dir = home + args.out_dir
# ======================================================================================
# Load Params
# ======================================================================================


# ======================================================================================
# Load Corpus & Vocab
# ======================================================================================
corpus_file = args.corpus + "ptb.hdf5"
corpus_hdf5 = h5py.File(corpus_file, mode='r')

vocab = marisa_trie.Trie(corpus_hdf5["vocabulary"])
vocab_size = len(vocab)
print("Vocabulary loaded: {} words".format(vocab_size))

# ======================================================================================
# Build Model
# ======================================================================================
print(args)

# N-Gram size should also be verified against dataset attributes
model = NNLM(ngram_size=args.ngram_size - 1,
             vocab_size=vocab_size,
             embed_dim=args.embed_dim,
             batch_size=args.batch_size,
             h_dim=args.h_dim)

labels = Input(n_units=len(vocab), name="word_classes")
loss = model.loss(labels.tensor)

optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
train_step = optimizer.minimize(loss)

# ======================================================================================
# Training
# ======================================================================================
print("starting TF")
tf_var_init = tf.global_variables_initializer()
sess_config = tf.ConfigProto(intra_op_parallelism_threads=8)
sess = tf.Session(config=sess_config)
sess.run(tf_var_init)

ngrams_processed = 0

for epoch in range(args.epochs):
    # restart training dataset
    # TODO shuffle the data ?
    # train_dataset = ptb_reader.training_set(n_samples=50)
    # ngram_stream = (ngram_windows(sentence, args.window_size) for sentence in train_dataset)
    training_dataset = corpus_hdf5["training"]
    print(training_dataset[0:10])
    ngram_stream = chunk_it(training_dataset, chunk_size=args.batch_size * 100)
    # load batches
    x_batch = []
    y_batch = []
    b = 0

    # stream of ngrams for each sentence
    for ngram in ngram_stream:
        # wi hi already come as indices of the words they represent
        # print("ngram: ", ngram)
        # print(list(map(vocab.restore_key,ngram)))
        wi = ngram[-1]
        hi = ngram[:-1]
        # x_batch.append(list(map(vocab.get, hi)))
        x_batch.append(hi)
        # one hot
        y = np.zeros([vocab_size])
        # y[vocab[wi]] = 1
        y[wi] = 1
        y_batch.append(y)

        b += 1
        ngrams_processed += 1

        if b % args.batch_size == 0:
            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)
            # train the model

            current_loss = sess.run(loss, {
                model.inputs.tensor: x_batch,
                labels.tensor: y_batch,
            })
            # print("loss before: ", current_loss)
            # train step
            sess.run(train_step, {
                model.inputs.tensor: x_batch,
                labels.tensor: y_batch,
            })
            current_loss = sess.run(loss, {
                model.inputs.tensor: x_batch,
                labels.tensor: y_batch,
            })
            # print("loss after: ", current_loss)

            x_batch = []
            y_batch = []

    print("Processed {} ngrams".format(ngrams_processed))
