import argparse
import csv
import marisa_trie
import os

import h5py
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import tensorx as tx
from deepsign.data import transform
from deepsign.data.views import chunk_it, batch_it, shuffle_it, repeat_fn, take_it
from deepsign.models.models import NNLM
from tensorx.layers import Input

# ======================================================================================
# ARGUMENTS
#
# -conf : configuration file path
# -corpus : dataset file path (uses the hdf5 format defined by convert to hdf5 script)
# ======================================================================================
home = os.getenv("HOME")
default_out_dir = os.getcwd()
default_corpus = os.path.join(home, "data/datasets/ptb/")

parser = argparse.ArgumentParser(description="NNLM Baseline Parameters")
# prefix used to identify result files
parser.add_argument('-id', dest="id", type=int, default=0)
parser.add_argument('-corpus', dest="corpus", type=str, default=default_corpus)
parser.add_argument('-out_dir', dest="out_dir", type=str, default=default_out_dir)
parser.add_argument('-embed_dim', dest="embed_dim", type=int, default=64)
parser.add_argument('-embed_init', dest="embed_init", type=str, choices=["normal", "uniform"], default="uniform")
parser.add_argument('-logit_init', dest="logit_init", type=str, choices=["normal", "uniform"], default="uniform")
parser.add_argument('-embed_limits', dest="embed_limits", type=float, default=0.01)
parser.add_argument('-logit_limits', dest="logit_limits", type=float, default=0.01)
parser.add_argument('-h_dim', dest="h_dim", type=int, default=128)
parser.add_argument('-h_act', dest="h_act", type=str, choices=['relu', 'tanh', "elu"], default="relu")
parser.add_argument('-num_h', dest="num_h", type=int, default=2)
parser.add_argument('-shuffle', dest="shuffle", type=bool, default=True)
parser.add_argument('-shuffle_buffer_size', dest="shuffle_buffer_size", type=int, default=100 * 12800)
parser.add_argument('-epochs', dest="epochs", type=int, default=4)
parser.add_argument('-ngram_size', dest="ngram_size", type=int, default=4)
parser.add_argument('-batch_size', dest="batch_size", type=int, default=128)
parser.add_argument('-clip_gradients', dest="clip_gradients", type=bool, default=False)
parser.add_argument('-clip_norm', dest="clip_norm", type=float, default=1.0)
parser.add_argument('-learning_rate', dest="learning_rate", type=float, default=0.001)
parser.add_argument('-optimizer', dest="optimizer", type=str, choices=["sgd", "ams"], default="sgd")

# only needed for adam and ams
parser.add_argument('-optimizer_beta1', dest="optimizer_beta1", type=float, default=0.9)
parser.add_argument('-optimizer_beta2', dest="optimizer_beta2", type=float, default=0.999)
parser.add_argument('-optimizer_epsilon', dest="optimizer_epsilon", type=float, default=1e-8)

# model checkpoint frequency
parser.add_argument('-model_eval_checkpoint', dest='model_eval_checkpoint', type=bool, default=False)

parser.add_argument('-lr_decay', dest='lr_decay', type=bool, default=True)
parser.add_argument('-lr_decay_rate', dest='lr_decay_rate', type=float, default=1)
parser.add_argument('-lr_decay_on_eval', dest='lr_decay_on_eval', type=bool, default=True)
parser.add_argument('-dropout', dest='dropout', type=bool, default=True)
parser.add_argument('-keep_prob', dest='keep_prob', type=float, default=0.95)
args = parser.parse_args()
# ======================================================================================
# Load Params, Prepare results files
# ======================================================================================


# ======================================================================================
# Load Corpus & Vocab
# ======================================================================================
corpus_file = os.path.join(args.corpus, "ptb.hdf5")
corpus_hdf5 = h5py.File(corpus_file, mode='r')

vocab = marisa_trie.Trie(corpus_hdf5["vocabulary"])
vocab_size = len(vocab)
print("Vocabulary loaded: {} words".format(vocab_size))

# corpus
training_dataset = corpus_hdf5["training"]
test_dataset = corpus_hdf5["test"]
validation_dataset = corpus_hdf5["validation"]


# data pipeline
def data_pipeline(hdf5_dataset, epochs=1, batch_size=args.batch_size, shuffle=args.shuffle):
    def chunk_fn(x):
        return chunk_it(x, chunk_size=batch_size * 1000)

    if epochs > 1:
        dataset = repeat_fn(chunk_fn, hdf5_dataset, epochs)
    else:
        dataset = chunk_fn(hdf5_dataset)

    if shuffle:
        dataset = shuffle_it(dataset, args.shuffle_buffer_size)

    # cannot pad because 0 might be a valid index and that screws our evaluation
    # padding = np.zeros([args.ngram_size], dtype=np.int64)
    # dataset = batch_it(dataset, size=batch_size, padding=True, padding_elem=padding)
    dataset = batch_it(dataset, size=batch_size, padding=False)
    return dataset


# ======================================================================================
# MODEL
# ======================================================================================

# Hidden layer weight init.
if args.h_act == "relu":
    h_act = tx.relu
    h_init = tx.he_normal_init()
elif args.h_act == "elu":
    h_act = tx.elu
    h_init = tx.he_normal_init()
elif args.h_act == "tanh":
    h_act = tx.tanh
    h_init = tx.xavier_init()

# Embedding (Lookup) layer weight init
if args.embed_init == "normal":
    embed_init = tx.random_normal(mean=0, stdev=args.embed_limits)
elif args.embed_init == "uniform":
    embed_init = tx.random_uniform(maxval=args.embed_limits, minval=-args.embed_limits)

# Logit layer weight init
if args.logit_init == "normal":
    logit_init = embed_init = tx.random_normal(mean=0, stdev=args.embed_limits)
elif args.logit_init == "uniform":
    logit_init = tx.random_uniform(maxval=args.embed_limits, minval=-args.embed_limits)

model = NNLM(ctx_size=args.ngram_size - 1,
             vocab_size=vocab_size,
             embed_dim=args.embed_dim,
             embed_init=embed_init,
             logit_init=logit_init,
             batch_size=args.batch_size,
             h_dim=args.h_dim,
             num_h=args.num_h,
             h_activation=h_act,
             h_init=h_init,
             use_dropout=args.dropout,
             keep_prob=args.keep_prob,
             nce=False,
             nce_samples=100)

model_runner = tx.ModelRunner(model)

lr_param = tx.InputParam(init_value=args.learning_rate)
# optimizer = tf.train.AdamOptimizer(learning_rate=lr_param.tensor)

# optimizer = tf.train.RMSPropOptimizer(learning_rate=args.learning_rate)


# optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr_param.tensor)
optimizer = tx.AMSGrad(learning_rate=lr_param.tensor)


# optimizer = tx.AMSGrad(learning_rate=lr_param.tensor,
#                           beta1=args.optimizer_beta1,
#                           beta2=args.optimizer_beta2,
#                           epsilon=args.optimizer_epsilon)


# model_runner.config_optimizer(optimizer)
def global_grad_clip(grads):
    grads, _ = tf.clip_by_global_norm(grads, args.clip_norm)
    return grads


def local_grad_clip(grad):
    return tf.clip_by_norm(grad, args.clip_norm)


if args.clip_gradients:
    model_runner.config_optimizer(optimizer, params=lr_param, gradient_op=local_grad_clip, global_gradient_op=False)
else:
    model_runner.config_optimizer(optimizer, params=lr_param)

sess = tf.Session()
model_runner.set_session(sess)


# ======================================================================================
# EVALUATION UTIL FUNCTIONS
# ======================================================================================
def eval_model(runner, dataset_it, len_dataset=None):
    pb = tqdm(total=len_dataset, ncols=60)
    batches_processed = 0
    sum_loss = 0
    for batch in dataset_it:
        batch = np.array(batch, dtype=np.int64)
        ctx = batch[:, :-1]
        target = batch[:, -1:]

        mean_loss = runner.eval(ctx, target)
        # print(mean_loss)
        sum_loss += mean_loss

        pb.update(args.batch_size)
        batches_processed += 1

    pb.close()

    return np.exp(sum_loss / batches_processed)


# ======================================================================================
# TRAINING LOOP
# ======================================================================================
print("Starting Training")

# preparing evaluation steps
# I use ceil because I make sure we have padded batches at the end
num_batches = np.ceil(len(training_dataset) / args.batch_size)
eval_step = num_batches // 2
print("eval every ", eval_step)

epoch_step = 0
global_step = 0
current_epoch = 0
current_lr = args.learning_rate

model_runner.init_vars()
progress = tqdm(total=len(training_dataset) * args.epochs)
training_data = data_pipeline(training_dataset, epochs=args.epochs)

#ppl = eval_model(model_runner, data_pipeline(validation_dataset, epochs=1, shuffle=False), len(validation_dataset))
#progress.write("val perplexity {}".format(ppl))

for ngram_batch in training_data:
    epoch = progress.n // len(training_dataset) + 1
    # ================================================
    # CHANGING EPOCH restart step
    # ================================================
    if epoch != current_epoch:
        current_epoch = epoch
        epoch_step = 0
        progress.write("epoch: {}".format(current_epoch))
        if epoch > 1:
            ppl = eval_model(model_runner, data_pipeline(validation_dataset), len(validation_dataset))
            progress.write("val perplexity {}".format(ppl))

    ngram_batch = np.array(ngram_batch, dtype=np.int64)
    ctx_ids = ngram_batch[:, :-1]
    word_ids = ngram_batch[:, -1:]

    assert (np.ndim(word_ids) == 2)
    model_runner.train(ctx_ids, word_ids)
    progress.update(args.batch_size)

    epoch_step += 1

    if epoch_step % eval_step == 0 and epoch_step > 0:
        ppl = eval_model(model_runner, data_pipeline(validation_dataset), len(validation_dataset))
        progress.write("val perplexity {}".format(ppl))

model_runner.close_session()
progress.close()
