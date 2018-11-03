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
from deepsign.data.views import chunk_it, batch_it, shuffle_it, repeat_apply
from deepsign.models.nrp import LBL_NRP
from tensorx.layers import Input

from deepsign.rp.ri import Generator, RandomIndex
from deepsign.rp.tf_utils import ris_to_sp_tensor_value


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

param("k_dim", int, 4000)
param("s_active", int, 4)

param("embed_dim", int, 128)

param("embed_init", str, "uniform", valid=["normal", "uniform"])
param("embed_init_val", float, 0.01)

param("logit_init", str, "uniform", valid=["normal", "uniform"])
param("logit_init_val", float, 0.01)

param("use_gate", str2bool, True)
param("use_hidden", str2bool, False)
param("embed_share", str2bool, True)

param("x_to_f_init", str, "uniform", valid=["normal", "uniform"])
param("x_to_f_init_val", float, 0.01)
param("h_to_f_init", str, "uniform", valid=["normal", "uniform"])
param("h_to_f_init_val", float, 0.01)

param("num_h", int, 1)
param("h_dim", int, 256)
param("h_act", str, "relu", valid=['relu', 'tanh', 'elu'])

param("epochs", int, 4)
param("batch_size", int, 128)
param("shuffle", str2bool, True)
param("shuffle_buffer_size", int, 128 * 10000)

param("optimizer", str, "sgd", valid=["sgd", "adam", "ams"])
# only needed for adam and ams
param("optimizer_beta1", float, 0.9)
param("optimizer_beta2", float, 0.999)
param("optimizer_epsilon", float, 1e-8)

param("lr", float, 0.5)
param("lr_decay", str2bool, False)
param("lr_decay_rate", float, 0.5)
# lr does not decay beyond this threshold
param("lr_decay_threshold", float, 1e-6)
# lr decay when last_ppl - current_ppl < eval_threshold
param("eval_threshold", float, 1.0)

# number of epochs without improvement before stopping
param("early_stop", str2bool, True)
param("patience", int, 3)
param("use_f_predict", str2bool, False)

# REGULARISATION
# clip grads by norm
param("clip_grads", str2bool, True)
# if true clips by local norm, else clip by norm of all gradients
param("clip_local", str2bool, False)
param("clip_value", float, 1.0)

param("dropout", str2bool, True)
param("embed_dropout", str2bool, True)
param("keep_prob", float, 0.95)

param("l2_loss", str2bool, False)
param("l2_loss_coef", float, 1e-5)

args = parser.parse_args()
# ======================================================================================
# Load Params, Prepare results assets
# ======================================================================================

# Experiment parameter summary
res_param_filename = os.path.join(args.out_dir, "params_{id}.csv".format(id=args.id))
with open(res_param_filename, "w") as param_file:
    arg_dict = vars(args)
    writer = csv.DictWriter(f=param_file, fieldnames=arg_dict.keys())
    writer.writeheader()
    writer.writerow(arg_dict)
    param_file.flush()

# make dir for model checkpoints
if args.save_model:
    model_ckpt_dir = os.path.join(args.out_dir, "model_{id}".format(id=args.id))
    os.makedirs(model_ckpt_dir, exist_ok=True)
    model_path = os.path.join(model_ckpt_dir, "nnlm_{id}.ckpt".format(id=args.id))

# start perplexity file
ppl_header = ["id", "epoch", "step", "lr", "dataset", "perplexity"]
ppl_fname = os.path.join(args.out_dir, "perplexity_{id}.csv".format(id=args.id))

ppl_file = open(ppl_fname, "w")
ppl_writer = csv.DictWriter(f=ppl_file, fieldnames=ppl_header)
ppl_writer.writeheader()

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
ri_tensor = ris_to_sp_tensor_value(ris, dim=args.k_dim)

print("done")


# ======================================================================================

def data_pipeline(data, epochs=1, batch_size=args.batch_size, shuffle=False):
    def chunk_fn(x):
        return chunk_it(x, chunk_size=batch_size * 1000)

    if epochs > 1:
        data = repeat_apply(chunk_fn, data, epochs)
    else:
        data = chunk_fn(data)

    if shuffle:
        data = shuffle_it(data, args.shuffle_buffer_size)

    data = batch_it(data, size=batch_size, padding=False)
    return data


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

if args.h_to_f_init == "normal":
    h_to_f_init = tx.random_normal(mean=0., stddev=args.h_to_f_init_val)
elif args.h_to_f_init == "uniform":
    h_to_f_init = tx.random_uniform(minval=-args.h_to_f_init_val, maxval=args.h_to_f_init_val)

if args.x_to_f_init == "normal":
    x_to_f_init = tx.random_normal(mean=0., stddev=args.x_to_f_init_val)
elif args.h_to_f_init == "uniform":
    x_to_f_init = tx.random_uniform(minval=-args.h_to_f_init_val, maxval=args.x_to_f_init_val)

model = LBL_NRP(ctx_size=args.ngram_size - 1,
                vocab_size=len(vocab),
                k_dim=args.k_dim,
                ri_tensor=ri_tensor,
                embed_dim=args.embed_dim,
                embed_init=embed_init,
                x_to_f_init=x_to_f_init,
                logit_init=logit_init,
                embed_share=args.embed_share,
                use_gate=args.use_gate,
                use_hidden=args.use_hidden,
                h_dim=args.h_dim,
                h_activation=h_act,
                h_init=h_init,
                h_to_f_init=h_to_f_init,
                use_dropout=args.dropout,
                embed_dropout=args.embed_dropout,
                keep_prob=args.keep_prob,
                l2_loss=args.l2_loss,
                l2_loss_coef=args.l2_loss_coef
                )

model_runner = tx.ModelRunner(model)

# we use an InputParam because we might want to change it during training
lr_param = tx.InputParam(init_value=args.lr)
if args.optimizer == "sgd":
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr_param.tensor)
elif args.optimizer == "adam":
    optimizer = tf.train.AdamOptimizer(learning_rate=lr_param.tensor,
                                       beta1=args.optimizer_beta1,
                                       beta2=args.optimizer_beta2,
                                       epsilon=args.optimizer_epsilon)
elif args.optimizer == "ams":
    optimizer = tx.AMSGrad(learning_rate=lr_param.tensor,
                           beta1=args.optimizer_beta1,
                           beta2=args.optimizer_beta2,
                           epsilon=args.optimizer_epsilon)


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


# ======================================================================================
# EVALUATION
# ======================================================================================


def eval_model(runner, dataset_it, len_dataset=None, display_progress=False):
    if display_progress:
        pb = tqdm(total=len_dataset, ncols=60)
    batches_processed = 0
    sum_loss = 0
    for batch in dataset_it:
        batch = np.array(batch, dtype=np.int64)
        ctx = batch[:, :-1]
        target = batch[:, -1:]

        mean_loss = runner.eval(ctx, target)
        sum_loss += mean_loss

        if display_progress:
            pb.update(args.batch_size)
        batches_processed += 1

    if display_progress:
        pb.close()

    return np.exp(sum_loss / batches_processed)


def evaluation(runner: tx.ModelRunner, pb, cur_epoch, step, display_progress=False):
    pb.write("[Eval Validation]")

    val_data = corpus["validation"]
    ppl_validation = eval_model(runner, data_pipeline(val_data, epochs=1, shuffle=False), len(val_data),
                                display_progress)
    res_row = {"id": args.id, "epoch": cur_epoch, "step": step, "lr": lr_param.value, "dataset": "validation",
               "perplexity": ppl_validation}
    ppl_writer.writerow(res_row)

    pb.write("Eval Test")
    test_data = corpus["test"]
    ppl_test = eval_model(runner, data_pipeline(test_data, epochs=1, shuffle=False), len(test_data), display_progress)

    res_row = {"id": args.id, "epoch": cur_epoch, "step": step, "lr": lr_param.value, "dataset": "test",
               "perplexity": ppl_test}
    ppl_writer.writerow(res_row)

    ppl_file.flush()

    pb.write("valid. ppl = {} \n test ppl {}".format(ppl_validation, ppl_test))
    return ppl_validation


# ======================================================================================
# TRAINING LOOP
# ======================================================================================
print("starting TF")

# preparing evaluation steps
# I use ceil because I make sure we have padded batches at the end

epoch_step = 0
global_step = 0
current_epoch = 0
patience = 0

sess = tf.Session()
model_runner.set_session(sess)
model_runner.init_vars()

training_dset = corpus["training"]
progress = tqdm(total=len(training_dset) * args.epochs)
training_data = data_pipeline(training_dset, epochs=args.epochs, shuffle=True)

evals = []

for ngram_batch in training_data:
    epoch = progress.n // len(training_dset) + 1
    # Start New Epoch
    if epoch != current_epoch:
        current_epoch = epoch
        epoch_step = 0
        progress.write("epoch: {}".format(current_epoch))

    # Eval Time
    if epoch_step == 0:
        current_eval = evaluation(model_runner, progress, epoch, global_step)
        evals.append(current_eval)

        if global_step > 0:
            if args.early_stop:
                if evals[-2] - evals[-1] < args.eval_threshold:
                    if patience >= 3:
                        progress.write("early stop")
                        break
                    patience += 1
                else:
                    patience = 0

            # lr decay only at the start of each epoch
            if args.lr_decay and len(evals) > 0:
                if evals[-2] - evals[-1] < args.eval_threshold:
                    lr_param.value = max(lr_param.value * args.lr_decay_rate, args.lr_decay_threshold)
                    progress.write("lr changed to {}".format(lr_param.value))

    # ================================================
    # TRAIN MODEL
    # ================================================
    ngram_batch = np.array(ngram_batch, dtype=np.int64)
    ctx_ids = ngram_batch[:, :-1]
    word_ids = ngram_batch[:, -1:]

    model_runner.train(ctx_ids, word_ids)
    progress.update(args.batch_size)

    epoch_step += 1
    global_step += 1

# if not early stop, evaluate last state of the model
if not args.early_stop or patience < 3:
    evaluation(model_runner, progress, epoch, epoch_step)
ppl_file.close()

if args.save_model:
    model_runner.save_model(model_name=model_path, step=global_step, write_state=False)

model_runner.close_session()
progress.close()
