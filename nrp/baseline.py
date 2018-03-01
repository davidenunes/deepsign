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
from deepsign.data.views import chunk_it, batch_it, shuffle_it, repeat_fn
from deepsign.models.models import NNLM
from tensorx.layers import Input


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
parser = argparse.ArgumentParser(description="NRP Baseline Experiment")
# experiment ID
parser.add_argument('-id', dest="id", type=int, default=0)

# corpus and ngram size should match since we pre-process the corpus to yield n-grams
default_corpus = os.path.join(os.getenv("HOME"), "data/datasets/ptb/")
parser.add_argument('-corpus', dest="corpus", type=str, default=default_corpus)
parser.add_argument('-ngram_size', dest="ngram_size", type=int, default=4)

default_out_dir = os.getcwd()
parser.add_argument('-save_model', dest='save_model', type=str2bool, default=False)
parser.add_argument('-out_dir', dest="out_dir", type=str, default=default_out_dir)

parser.add_argument('-embed_dim', dest="embed_dim", type=int, default=64)
parser.add_argument('-embed_init', dest="embed_init", type=str, choices=["normal", "uniform"], default="normal")
parser.add_argument('-embed_init_val', dest="embed_init_val", type=float, default=0.01)
parser.add_argument('-logit_init', dest="logit_init", type=str, choices=["normal", "uniform"], default="normal")
parser.add_argument('-logit_init_val', dest="logit_init_val", type=float, default=0.01)

parser.add_argument('-h_dim', dest="h_dim", type=int, default=128)
parser.add_argument('-h_act', dest="h_act", type=str, choices=['relu', 'tanh', 'elu'], default="elu")
parser.add_argument('-num_h', dest="num_h", type=int, default=1)

# training data pipeline
parser.add_argument('-epochs', dest="epochs", type=int, default=2)
parser.add_argument('-shuffle', dest="shuffle", type=str2bool, default=True)
parser.add_argument('-shuffle_buffer_size', dest="shuffle_buffer_size", type=int, default=128 * 100000)
parser.add_argument('-batch_size', dest="batch_size", type=int, default=128)

parser.add_argument('-optimizer', dest="optimizer", type=str, choices=["sgd", "adam", "ams"], default="ams")
# only needed for adam and ams
parser.add_argument('-optimizer_beta1', dest="optimizer_beta1", type=float, default=0.9)
parser.add_argument('-optimizer_beta2', dest="optimizer_beta2", type=float, default=0.999)
parser.add_argument('-optimizer_epsilon', dest="optimizer_epsilon", type=float, default=1e-8)

parser.add_argument('-lr', dest="lr", type=float, default=0.001)
parser.add_argument('-lr_decay', dest='lr_decay', type=str2bool, default=False)
# lr does not decay beyond threshold
parser.add_argument('-lr_decay_threshold', dest='lr_decay_threshold', type=float, default=1e-5)
# lr decay when last_ppl - current_ppl < eval_threshold
parser.add_argument('-eval_threshold', dest='eval_threshold', type=float, default=1.0)
parser.add_argument('-lr_decay_rate', dest='lr_decay_rate', type=float, default=0.5)
# number of epochs without improvement before stopping
parser.add_argument('-early_stop', dest='early_stop', type=str2bool, default=True)
parser.add_argument('-patience', dest='patience', type=int, default=3)

# REGULARISATION

# clip grads by norm
parser.add_argument('-clip_grads', dest="clip_grads", type=str2bool, default=True)
# if true clips each gradient by its norm, else clip all gradients by global norm
parser.add_argument('-clip_local', dest="clip_local", type=str2bool, default=True)
parser.add_argument('-clip_value', dest="clip_value", type=float, default=1.0)

# use dropout
parser.add_argument('-dropout', dest='dropout', type=str2bool, default=True)
parser.add_argument('-keep_prob', dest='keep_prob', type=float, default=0.9)
args = parser.parse_args()
# ======================================================================================
# Load Params, Prepare results files
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
# Load Corpus & Vocab
# ======================================================================================
corpus = h5py.File(os.path.join(args.corpus, "ptb.hdf5"), mode='r')
vocab = marisa_trie.Trie(corpus["vocabulary"])
vocab_size = len(vocab)


def data_pipeline(data, epochs=1, batch_size=args.batch_size, shuffle=False):
    def chunk_fn(x):
        return chunk_it(x, chunk_size=batch_size * 1000)

    if epochs > 1:
        data = repeat_fn(chunk_fn, data, epochs)
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
             keep_prob=args.keep_prob)

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
               "perplexity": ppl_validation}
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
