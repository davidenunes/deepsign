import argparse
import csv
import marisa_trie
import os

import h5py
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import tensorx as tx
from deepsign.data.views import chunk_it, batch_it, shuffle_it, repeat_apply, take_it
from deepsign.models.nrp import NNLM_NRP, RandomIndexTensor

from deepsign.rp.ri import Generator, RandomIndex
from deepsign.rp.tf_utils import to_sparse_tensor_value


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# ======================================================================================
# Experiment Args
# ======================================================================================
parser = argparse.ArgumentParser(description="LBL base experiment")


# clean argparse a bit
def param(name, argtype, default, valid=None):
    if valid is not None:
        parser.add_argument('-{}'.format(name), dest=name, type=argtype, default=default, choices=valid)
    else:
        parser.add_argument('-{}'.format(name), dest=name, type=argtype, default=default)


default_corpus = os.path.join(os.getenv("HOME"), "data/datasets/ptb/")
default_out_dir = os.getcwd()

# experiment ID
param("id", int, 0)
param("corpus", str, default_corpus)
param("ngram_size", int, 5)
param("save_model", str2bool, False)
param("out_dir", str, default_out_dir)

param("k_dim", int, 10000)
param("s_active", int, 16)

param("embed_dim", int, 64)

param("embed_init", str, "uniform", valid=["normal", "uniform"])
param("embed_init_val", float, 0.01)

param("logit_init", str, "uniform", valid=["normal", "uniform"])
param("logit_init_val", float, 0.01)

param("embed_share", str2bool, True)

param("num_h", int, 1)
param("h_dim", int, 128)
param("h_act", str, "relu", valid=['relu', 'tanh', 'elu'])

param("epochs", int, 2)
param("batch_size", int, 128)
param("shuffle", str2bool, True)
param("shuffle_buffer_size", int, 128 * 10000)

param("optimizer", str, "ams", valid=["sgd", "adam", "ams"])
# only needed for adam and ams
param("optimizer_beta1", float, 0.9)
param("optimizer_beta2", float, 0.999)
param("optimizer_epsilon", float, 1e-8)

param("lr", float, 0.5)
param("lr_decay", str2bool, True)
param("lr_decay_rate", float, 0.5)
# lr does not decay beyond this threshold
param("lr_decay_threshold", float, 1e-6)
# lr decay when last_ppl - current_ppl < eval_threshold
param("eval_threshold", float, 1.0)

# number of epochs without improvement before stopping
param("early_stop", str2bool, True)
param("patience", int, 3)
param("f_init", str, "uniform", valid=["normal", "uniform"])
param("f_init_val", float, 0.01)

# REGULARISATION
# clip grads by norm
param("clip_grads", str2bool, True)
# if true clips by local norm, else clip by norm of all gradients
param("clip_local", str2bool, True)
param("clip_value", float, 1.0)

param("dropout", str2bool, True)
param("embed_dropout", str2bool, True)
param("keep_prob", float, 0.85)

param("l2_loss", str2bool, False)
param("l2_loss_coef", float, 1e-5)

args = parser.parse_args()

# ======================================================================================
# CORPUS, Vocab and RIs
# ======================================================================================
corpus = h5py.File(os.path.join(args.corpus, "ptb_{}.hdf5".format(args.ngram_size)), mode='r')
vocab = marisa_trie.Trie(corpus["vocabulary"])

print("generating random indexes")
# generates k-dimensional random indexes with s_active units
ri_generator = Generator(dim=args.k_dim, num_active=args.s_active)

# pre-gen indices for vocab
# it doesn't matter which ri gets assign to which word since we are pre-generating the indexes
ris = [ri_generator.generate() for i in range(len(vocab))]

#ri_tensor = RandomIndexTensor.from_ri_list(ris, args.k_dim, args.s_active)
ri_tensor = to_sparse_tensor_value(ris, dim=args.k_dim)

print("done")
# ======================================================================================

# corpus
training_dataset = corpus["training"]
test_dataset = corpus["test"]
validation_dataset = corpus["validation"]


# data pipeline
def data_pipeline(hdf5_dataset, epochs=1, batch_size=args.batch_size, shuffle=args.shuffle):
    def chunk_fn(x):
        return chunk_it(x, chunk_size=batch_size * 1000)

    if epochs > 1:
        dataset = repeat_apply(chunk_fn, hdf5_dataset, epochs)
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
# Activation functions
if args.h_act == "relu":
    h_act = tx.relu
    h_init = tx.he_normal_init()
elif args.h_act == "tanh":
    h_act = tx.tanh
    h_init = tx.xavier_init()
elif args.h_act == "elu":
    h_act = tx.elu
    h_init = tx.he_normal_init()

# Parameter Init
if args.embed_init == "normal":
    embed_init = tx.random_normal(mean=0.,
                                  stddev=args.embed_init_val)
elif args.embed_init == "uniform":
    embed_init = tx.random_uniform(minval=-args.embed_init_val,
                                   maxval=args.embed_init_val)

if args.logit_init == "normal":
    logit_init = tx.random_normal(mean=0.,
                                  stddev=args.logit_init_val)
elif args.logit_init == "uniform":
    logit_init = tx.random_uniform(minval=-args.logit_init_val,
                                   maxval=args.logit_init_val)

if args.f_init == "normal":
    f_init = tx.random_normal(mean=0., stddev=args.f_init_val)
elif args.f_init == "uniform":
    f_init = tx.random_uniform(minval=-args.f_init_val, maxval=args.f_init_val)

model = NNLM_NRP(ctx_size=args.ngram_size - 1,
                 vocab_size=len(vocab),
                 k_dim=args.k_dim,
                 s_active=args.s_active,
                 ri_tensor=ri_tensor,
                 embed_dim=args.embed_dim,
                 embed_init=embed_init,
                 embed_share=args.embed_share,
                 logit_init=logit_init,
                 h_dim=args.h_dim,
                 num_h=args.num_h,
                 h_activation=h_act,
                 h_init=h_init,
                 use_dropout=args.dropout,
                 keep_prob=args.keep_prob,
                 embed_dropout=args.embed_dropout,
                 l2_loss=args.l2_loss,
                 l2_loss_coef=args.l2_loss_coef,
                 f_init=f_init,
                 logit_bias=False)

model_runner = tx.ModelRunner(model)

lr_param = tx.InputParam(init_value=args.lr)
# optimizer = tf.train.AdamOptimizer(learning_rate=lr_param.tensor)

# optimizer = tf.train.RMSPropOptimizer(learning_rate=args.learning_rate)


# optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr_param.tensor)
optimizer = tx.AMSGrad(learning_rate=lr_param.tensor)


# optimizer = tx.AMSGrad(learning_rate=lr_param.tensor,
#                           beta1=args.optimizer_beta1,
#                           beta2=args.optimizer_beta2,
#                           epsilon=args.optimizer_epsilon)


def clip_grad_global(grads):
    grads, _ = tf.clip_by_global_norm(grads, 12)
    return grads


def clip_grad_local(grad):
    return tf.clip_by_norm(grad, args.clip_value)


if args.clip_grads:
    if args.clip_local:
        clip_fn = clip_grad_local
    else:
        clip_fn = clip_grad_global

if args.clip_grads:
    model_runner.config_optimizer(optimizer, params=lr_param,
                                  gradient_op=clip_fn,
                                  global_gradient_op=not args.clip_local)
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
current_lr = args.lr

model_runner.init_vars()
progress = tqdm(total=len(training_dataset) * args.epochs)
training_data = data_pipeline(training_dataset, epochs=args.epochs)

# first eval just to make sure it is something like 9999
ppl = eval_model(model_runner, data_pipeline(validation_dataset, epochs=1, shuffle=False), len(validation_dataset))
progress.write("val perplexity {}".format(ppl))

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
