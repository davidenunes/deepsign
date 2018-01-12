import argparse
import os
import h5py
import marisa_trie
from deepsign.models.nnlm import NNLM

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from deepsign.data.views import chunk_it, batch_it, shuffle_it, repeat_it, take_it
from tensorx.layers import Input
import tensorx as tx

from deepsign.data import transform

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
parser.add_argument('-batch_size', dest="batch_size", type=int, default=512)
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

# corpus
training_dataset = corpus_hdf5["training"]
chunk_size = args.batch_size * 100
ngram_stream = chunk_it(training_dataset, chunk_size=chunk_size)

ngram_stream = repeat_it(ngram_stream, args.epochs)

padding = np.zeros([args.ngram_size],dtype=np.int64)
# batch n-grams
ngram_stream = batch_it(ngram_stream, size=args.batch_size, padding=True, padding_elem=padding)

# ======================================================================================
# Build Model
# ======================================================================================

# N-Gram size should also be verified against dataset attributes
inputs = Input(n_units=args.ngram_size - 1, name="context_indices", dtype=tf.int64)
loss_inputs = Input(n_units=vocab_size, batch_size=args.batch_size, dtype=tf.int64)

model = NNLM(inputs=inputs, loss_inputs=loss_inputs,
             n_gram_size=args.ngram_size - 1,
             vocab_size=vocab_size,
             embed_dim=args.embed_dim,
             batch_size=args.batch_size,
             h_dim=args.h_dim)

model_runner = tx.ModelRunner(model)

optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
model_runner.config_training(optimizer)

# sess_config = tf.ConfigProto(intra_op_parallelism_threads=8)
# sess = tf.Session(config=sess_config)
sess = tf.Session()
model_runner.set_session(sess)

# ======================================================================================
# Training
# ======================================================================================
print("starting TF")

ngrams_processed = 0
progress = tqdm(total=len(training_dataset) * args.epochs)
#ngram_stream = take_it(1, ngram_stream)
for ngram_batch in ngram_stream:
    ngram_batch = np.array(ngram_batch,dtype=np.int64)
    ctx_ids = ngram_batch[:, :-1]
    word_ids = ngram_batch[:, -1]
    one_hot = transform.batch_one_hot(word_ids, vocab_size)

    model_runner.run(ctx_ids)
    progress.update(args.batch_size)
    """
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

        model_runner.train(data=x_batch, loss_input_data=y_batch)
        result = model_runner.run(x_batch)
        # print(result)

        x_batch = []
        y_batch = []
    """
progress.close()
print("Processed {} ngrams".format(ngrams_processed))
