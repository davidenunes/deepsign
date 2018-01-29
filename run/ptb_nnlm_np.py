import argparse
import os
import h5py
import marisa_trie
from deepsign.models.nnlm import NNLM
import csv

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from deepsign.data.views import chunk_it, batch_it, shuffle_it, repeat_fn, take_it
from tensorx.layers import Input
import tensorx as tx

from deepsign.data import transform

# ======================================================================================
# ARGUMENTS
#
# -conf : configuration file path
# -corpus : dataset file path (uses the hdf5 format defined by convert to hdf5 script)
# -output_dir : output_dir dir where results are written, defaults to ~/data/results/
# ======================================================================================
home = os.getenv("HOME")

parser = argparse.ArgumentParser(description="NNLM Baseline Parameters")
# prefix used to identify result files
parser.add_argument('-id', dest="id", type=int, default=0)
parser.add_argument('-conf', dest="conf", type=str)
parser.add_argument('-corpus', dest="corpus", type=str, default=home + "/data/datasets/ptb/")
parser.add_argument('-output_dir', dest="output_dir", type=str, default=home + "/data/results/")
parser.add_argument('-embed_dim', dest="embed_dim", type=int, default=100)
parser.add_argument('-h_dim', dest="h_dim", type=int, default=200)
parser.add_argument('-epochs', dest="epochs", type=int, default=1)
parser.add_argument('-ngram_size', dest="ngram_size", type=int, default=4)
parser.add_argument('-out_dir', dest="out_dir", type=str, default="/data/results/")
parser.add_argument('-data_dir', dest="data_dir", type=str, default="/data/gold_standards/")
parser.add_argument('-learning_rate', dest="learning_rate", type=float, default=0.01)
parser.add_argument('-batch_size', dest="batch_size", type=int, default=128)
parser.add_argument('-eval_per_epoch', dest='eval_per_epoch', type=int, default=1)
args = parser.parse_args()

out_dir = home + args.out_dir
# ======================================================================================
# Load Params
# ======================================================================================
arg_dict = vars(args)

# result file name
print(arg_dict)
res_param_filename = "{id}_params.csv".format(id=arg_dict["id"])
with open(res_param_filename, "w") as param_file:
    writer = csv.DictWriter(f=param_file, fieldnames=arg_dict.keys())
    writer.writeheader()
    writer.writerow(arg_dict)

res_eval_filename = "{id}_eval.csv".format(id=arg_dict["id"])
eval_header = ["epoch", "step",
               "validation_avg_ppl",
               "test_avg_ppl"]

res_eval_file = open(res_eval_filename, "w")
res_eval_writer = csv.DictWriter(f=res_eval_file, fieldnames=eval_header)
res_eval_writer.writeheader()

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
test_dataset = corpus_hdf5["test"]
validation_dataset = corpus_hdf5["validation"]


# data pipeline
def get_data_it(hdf5_dataset):
    def chunk_fn(x): return chunk_it(x, chunk_size=args.batch_size * 100)

    dataset = repeat_fn(chunk_fn, hdf5_dataset, args.epochs)
    padding = np.zeros([args.ngram_size], dtype=np.int64)
    dataset = batch_it(dataset, size=args.batch_size, padding=True, padding_elem=padding)
    return dataset


# ======================================================================================
# MODEL
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

# optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
optimizer = tx.AMSGrad(learning_rate=args.learning_rate)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate)

model_runner.config_optimizer(optimizer)

# sess_config = tf.ConfigProto(intra_op_parallelism_threads=8)
# sess = tf.Session(config=sess_config)
sess = tf.Session()
model_runner.set_session(sess)


# ======================================================================================
# EVAL
# ======================================================================================
def evaluation(model_runner, dataset_it, len_dataset):
    progress = tqdm(total=len_dataset)
    ngrams_processed = 0
    sum_loss = 0
    for ngram_batch in dataset_it:
        ngram_batch = np.array(ngram_batch, dtype=np.int64)
        ctx_ids = ngram_batch[:, :-1]
        word_ids = ngram_batch[:, -1]
        one_hot = transform.batch_one_hot(word_ids, vocab_size)

        mean_loss = model_runner.eval(ctx_ids, one_hot)
        sum_loss += mean_loss

        progress.update(args.batch_size)
        ngrams_processed += 1

    progress.close()
    return np.exp(sum_loss / ngrams_processed)


# ======================================================================================
# Training
# ======================================================================================
print("starting TF")

eval_points = np.linspace(0, 1, args.eval_per_epoch)
print("eval points", eval_points)
steps = 0
current_epoch = 0
progress = tqdm(total=len(training_dataset) * args.epochs)
# ngram_stream = take_it(1, ngram_stream)
training_data = get_data_it(training_dataset)
for ngram_batch in training_data:
    epoch = progress.n // len(training_dataset) + 1
    if epoch != current_epoch:
        current_epoch = epoch
        print("epoch: ", current_epoch)

    ngram_batch = np.array(ngram_batch, dtype=np.int64)
    ctx_ids = ngram_batch[:, :-1]
    word_ids = ngram_batch[:, -1]
    one_hot = transform.batch_one_hot(word_ids, vocab_size)

    model_runner.train(ctx_ids, one_hot)
    progress.update(args.batch_size)

    steps += 1
    if steps % 100 == 0:
        print("evaluating validation dataset")
        ppl_validation = evaluation(model_runner, get_data_it(validation_dataset), len(validation_dataset))

        print("evaluating test dataset")
        ppl_test = evaluation(model_runner, get_data_it(test_dataset), len(test_dataset))
        print("valid. perplexity = {} \n test perplexity {}".format(ppl_validation, ppl_test))

        res = {"epoch": epoch, "step": steps, "validation_avg_ppl": ppl_validation, "test_avg_ppl": ppl_test}
        res_eval_writer.writerow(res)
        res_eval_file.flush()

        print("results written")

    # print(epoch)

model_runner.close_session()
print("Processed {} ngrams".format(progress.n))
progress.close()

# close result files
res_eval_file.close()
