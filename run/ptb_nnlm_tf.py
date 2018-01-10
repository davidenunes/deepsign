import argparse
import os
import h5py
import marisa_trie
from deepsign.models.nnlm import NNLM

import numpy as np
import tensorflow as tf

from deepsign.data.views import chunk_it
import tensorx as tx
from tqdm import tqdm

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
parser.add_argument('-h_dim', dest="h_dim", type=int, default=10)
parser.add_argument('-epochs', dest="epochs", type=int, default=1)
parser.add_argument('-ngram_size', dest="ngram_size", type=int, default=4)
parser.add_argument('-out_dir', dest="out_dir", type=str, default="/data/results/")
parser.add_argument('-data_dir', dest="data_dir", type=str, default="/data/gold_standards/")
parser.add_argument('-learning_rate', dest="learning_rate", type=float, default=0.01)
parser.add_argument('-batch_size', dest="batch_size", type=int, default=50)
parser.add_argument('-n_rows', dest="n_rows", type=int, default=10)
parser.add_argument('-shuffle_buffer_size', dest="shuffle_buffer_size", type=int, default=1)
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

# load dataset into TF Dataset API
hdf5_training = corpus_hdf5["training"]
chunk_size = args.batch_size * args.shuffle_buffer_size



def n_gram_gen():
    """ Callable returning iterable over n-grams
    """
    for n_gram in chunk_it(hdf5_training, chunk_size=chunk_size):
        yield n_gram


training_data = tf.data.Dataset.from_generator(n_gram_gen, tf.int64)
if args.n_rows != -1:
    training_Data = training_data.take(args.n_rows)

training_data = training_data.repeat(args.epochs)
# training_data = training_data.prefetch(chunk_size*100)
training_data = training_data.shuffle(chunk_size)
training_data = training_data.batch(args.batch_size)


training_data_it = training_data.make_one_shot_iterator().get_next()

training_ctx, training_word = tf.split(training_data_it, [3, 1], axis=-1)
# [[2],[3]] -> [2,3]
training_word = tf.reshape(training_word, shape=[-1])

# example for dim=3 [0,1] -> [[1,0,0],[0,1,0]]
training_word_one_hot = tf.one_hot(training_word, depth=vocab_size)

# ======================================================================================
# Build Model
# ======================================================================================
print(args)

# N-Gram size should also be verified against dataset attributes
inputs = tx.TensorLayer(training_ctx, n_units=args.ngram_size - 1, batch_size=args.batch_size, dtype=tf.int64)
loss_inputs = tx.TensorLayer(training_word_one_hot, n_units=vocab_size, batch_size=args.batch_size, dtype=tf.int64)

model = NNLM(inputs=inputs, loss_inputs=loss_inputs,
             n_gram_size=args.ngram_size - 1,
             vocab_size=vocab_size,
             embed_dim=args.embed_dim,
             batch_size=args.batch_size,
             h_dim=args.h_dim)

model_runner = tx.ModelRunner(model)

optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
model_runner.config_training(optimizer)

#sess_config = tf.ConfigProto(intra_op_parallelism_threads=8)
#sess = tf.Session(config=sess_config)
sess = tf.Session()
model_runner.set_session(sess)

# ======================================================================================
# Training
# ======================================================================================
batches_processed = 0

total_n_grams = len(hdf5_training) * args.epochs
print(total_n_grams)
if args.n_rows != -1:
    total_n_grams = args.n_rows * args.epochs

progress = tqdm(total=total_n_grams/args.batch_size)
while True:
    try:
        ctx, one_hot_w = sess.run([training_ctx, training_word_one_hot])

        # the input to the model can be taken directly instead of using feed_dict
        # model_runner.train(data=ctx, loss_input_data=one_hot_w)
        model_runner.train()

        batches_processed += 1

        progress.update(args.batch_size)
        # print(batches_processed)

    except tf.errors.OutOfRangeError:
        print(np.shape(ctx))
        print(np.shape(one_hot_w))
        break
progress.close()

print("Processed {} batches".format(batches_processed))
