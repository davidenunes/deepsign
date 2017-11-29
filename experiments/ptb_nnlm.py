import argparse
import os
import sys
import csv
from collections import Counter
import marisa_trie
from deepsign.models.nnlm import NNLM

import h5py
import numpy as np
import tensorflow as tf
import tensorx as tx
from tqdm import tqdm

from deepsign.rp.index import TrieSignIndex
from deepsign.rp.ri import Generator
from deepsign.utils.views import chunk_it
from deepsign.utils.views import sliding_windows, ngram_windows
from tensorx.layers import Input
from deepsign.io.corpora.ptb import PTBReader

# ======================================================================================
# Argument parse configuration
# ======================================================================================
parser = argparse.ArgumentParser(description="NNLM Baseline Parameters")
parser.add_argument('-embed_dim', dest="embed_dim", type=int, default=100)
parser.add_argument('-h_dim', dest="h_dim", type=int, default=100)
parser.add_argument('-epochs', dest="epochs", type=int, default=1)
parser.add_argument('-window_size', dest="window_size", type=int, default=3)
parser.add_argument('-out_dir', dest="out_dir", type=str, default="/data/results/")
parser.add_argument('-data_dir', dest="data_dir", type=str, default="/data/datasets/")
parser.add_argument('-learning_rate', dest="learning_rate", type=float, default=0.01)
parser.add_argument('-batch_size', dest="batch_size", type=int, default=1)
args = parser.parse_args()

home = os.getenv("HOME")
out_dir = home + args.out_dir

# ======================================================================================
# Write Parameters
# ======================================================================================
args_filename = out_dir + "config.csv"
arg_dict = vars(args)
with open(args_filename, 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in arg_dict.items():
        writer.writerow([key, value])

        data_dir = home

# ======================================================================================
# Corpus
# ======================================================================================
corpus_dir = home + args.data_dir + "ptb"
ptb_reader = PTBReader(corpus_dir)

# load vocabulary
vocab_words = set()
for sentence in ptb_reader.full():
    vocab_words |= set(sentence)

# returns ids for words and words for ids with restore_key
# print(vocab.restore_key(9992))
vocab = marisa_trie.Trie(vocab_words)
print("Vocabulary loaded: {} words".format(len(vocab)))

# ======================================================================================
# Build Model
# ======================================================================================
model = NNLM(ngram_size=args.window_size - 1,
             vocab_size=len(vocab),
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
sess = tf.Session()
sess.run(tf_var_init)

ngrams_processed = 0

for epoch in range(args.epochs):
    # restart training dataset
    # TODO shuffle the data ?
    train_dataset = ptb_reader.training_set(n_samples=50)
    ngram_stream = (ngram_windows(sentence, args.window_size) for sentence in train_dataset)
    # load batches
    x_batch = []
    y_batch = []
    b = 0

    # stream of ngrams for each sentence
    for ngrams in ngram_stream:
        for ngram in ngrams:
            wi = ngram[-1]
            hi = ngram[:-1]
            x_batch.append(list(map(vocab.get, hi)))
            # one hot
            y = np.zeros([len(vocab)])
            y[vocab[wi]] = 1
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
                print("loss before: ", current_loss)
                # train step
                sess.run(train_step, {
                    model.inputs.tensor: x_batch,
                    labels.tensor: y_batch,
                })
                current_loss = sess.run(loss, {
                    model.inputs.tensor: x_batch,
                    labels.tensor: y_batch,
                })
                print("loss after: ", current_loss)

                x_batch = []
                y_batch = []

    print("Processed {} ngrams".format(ngrams_processed))