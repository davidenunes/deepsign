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
parser.add_argument('-embed_dim', dest="embed_dim", type=int, default=128)
parser.add_argument('-embed_init', dest="embed_init", type=str, choices=["normal", "uniform"], default="normal")
parser.add_argument('-logit_init', dest="logit_init", type=str, choices=["normal", "uniform"], default="normal")
parser.add_argument('-embed_limits', dest="embed_limits", type=float, default=0.01)
parser.add_argument('-logit_limits', dest="logit_limits", type=float, default=0.01)
parser.add_argument('-h_dim', dest="h_dim", type=int, default=256)
parser.add_argument('-h_act', dest="h_act", type=str, choices=['relu', 'tanh'], default="tanh")
parser.add_argument('-num_h', dest="num_h", type=int, default=1)
parser.add_argument('-shuffle', dest="shuffle", type=bool, default=True)
parser.add_argument('-shuffle_buffer_size', dest="shuffle_buffer_size", type=int, default=100 * 128)
parser.add_argument('-epochs', dest="epochs", type=int, default=4)
parser.add_argument('-ngram_size', dest="ngram_size", type=int, default=4)
parser.add_argument('-batch_size', dest="batch_size", type=int, default=128)
parser.add_argument('-clip_gradients', dest="clip_gradients", type=bool, default=True)
parser.add_argument('-clip_norm', dest="clip_norm", type=float, default=12.0)
# evaluation ratio size 0 < eval_epoch < 1 ex if 0.5 evals models on the middle of the dataset
parser.add_argument('-eval_step', dest='eval_step', type=float, default=0.5)
parser.add_argument('-learning_rate', dest="learning_rate", type=float, default=0.05)
parser.add_argument('-optimizer', dest="optimizer", type=str, choices=["sgd", "adam", "ams"], default="sgd")

# only needed for adam and ams
parser.add_argument('-optimizer_beta1', dest="optimizer_beta1", type=float, default=0.9)
parser.add_argument('-optimizer_beta2', dest="optimizer_beta2", type=float, default=0.999)
parser.add_argument('-optimizer_epsilon', dest="optimizer_epsilon", type=float, default=1e-8)

# model checkpoint frequency
parser.add_argument('-model_eval_checkpoint', dest='model_eval_checkpoint', type=bool, default=False)

parser.add_argument('-lr_decay', dest='lr_decay', type=bool, default=True)
parser.add_argument('-lr_decay_rate', dest='lr_decay_rate', type=float, default=0.5)
parser.add_argument('-lr_decay_on_eval', dest='lr_decay_on_eval', type=bool, default=True)
parser.add_argument('-dropout', dest='dropout', type=bool, default=True)
parser.add_argument('-keep_prob', dest='keep_prob', type=float, default=0.9)
args = parser.parse_args()
# ======================================================================================
# Load Params, Prepare results files
# ======================================================================================

# parameters file
res_param_filename = os.path.join(args.out_dir, "params_{id}.csv".format(id=args.id))
with open(res_param_filename, "w") as param_file:
    arg_dict = vars(args)
    writer = csv.DictWriter(f=param_file, fieldnames=arg_dict.keys())
    writer.writeheader()
    writer.writerow(arg_dict)
    param_file.flush()

# make dir for model checkpoints
model_ckpt_dir = os.path.join(args.out_dir, "model_{id}".format(id=args.id))
print(model_ckpt_dir)
os.makedirs(model_ckpt_dir, exist_ok=True)

model_path = os.path.join(model_ckpt_dir, "nnlm_{id}.ckpt".format(id=args.id))

# perplexity file
ppl_eval_filename = os.path.join(args.out_dir, "perplexity_{id}.csv".format(id=arg_dict["id"]))
eval_header = ["epoch", "step", "dataset", "perplexity"]

res_eval_file = open(ppl_eval_filename, "w")
res_eval_writer = csv.DictWriter(f=res_eval_file, fieldnames=eval_header)
res_eval_writer.writeheader()

"""
# dynamic hyperparams to be recorded
hyperparam_filename = os.path.join(args.out_dir, "hyperparams_{id}.csv".format(id=arg_dict["id"]))
hyperparam_header = ["epoch", "step", "param", "value"]
hyperparam_file = open(hyperparam_filename, "w")
hyperparam_writer = csv.DictWriter(f=hyperparam_file, fieldnames=hyperparam_header)
hyperparam_writer.writeheader()

def write_hyperparam(name, value, step, epoch):
    res_row = {"epoch": epoch, "step": step, "param": name, "value": value}
    hyperparam_writer.writerow(res_row)
    hyperparam_file.flush()

"""

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

# N-Gram size should also be verified against dataset attributes
inputs = Input(n_units=args.ngram_size - 1, name="context_indices", dtype=tf.int64)
loss_inputs = Input(n_units=vocab_size, batch_size=args.batch_size, dtype=tf.int64)

if args.h_act == "relu":
    h_act = tx.relu
    h_init = tx.relu_init()
if args.h_act == "tanh":
    h_act = tx.tanh
    h_init = tx.xavier_init()

if args.embed_init == "normal":
    embed_init = tx.random_normal(0, args.embed_limits)
elif args.embed_init == "uniform":
    embed_init = tx.random_uniform(0, args.embed_limits)
else:
    print(args.embed_init)
    raise ValueError("invalid embed_init, expected normal or uniform")

if args.logit_init == "normal":
    logit_init = tx.random_normal(0, args.logit_limits)
elif args.logit_init == "uniform":
    logit_init = tx.random_uniform(0, args.logit_limits)
else:
    print(args.logit_init)
    raise ValueError("invalid logit_init, expected normal or uniform")

model = NNLM(run_inputs=inputs, loss_inputs=loss_inputs,
             ctx_size=args.ngram_size - 1,
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

lr_param = tx.InputParam()
# optimizer = tf.train.AdamOptimizer(learning_rate=lr_param.tensor)

# optimizer = tx.AMSGrad(learning_rate=args.learning_rate)
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


# model_runner.config_optimizer(optimizer)
# def global_grad_clip(grads):
#    grads, _ = tf.clip_by_global_norm(grads, 12)
#    return grads

def clip_grad_norm(grad):
    return tf.clip_by_norm(grad, args.clip_norm)


if args.clip_gradients:
    model_runner.config_optimizer(optimizer, params=lr_param, gradient_op=clip_grad_norm, global_gradient_op=False)
else:
    model_runner.config_optimizer(optimizer, params=lr_param)

# sess_config = tf.ConfigProto(intra_op_parallelism_threads=8)
# sess = tf.Session(config=sess_config)
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
        target = batch[:, -1]
        target_one_hot = transform.batch_one_hot(target, vocab_size)

        mean_loss = runner.eval(ctx, target_one_hot)
        sum_loss += mean_loss

        pb.update(args.batch_size)
        batches_processed += 1

    pb.close()
    return np.exp(sum_loss / batches_processed)


def evaluation(runner: tx.ModelRunner, pb, epoch, step):
    pb.write("evaluating validation...")
    ppl_validation = eval_model(runner, data_pipeline(validation_dataset, epochs=1, shuffle=False),
                                len(validation_dataset))
    res_row = {"epoch": epoch, "step": step, "dataset": "validation", "perplexity": ppl_validation}
    res_eval_writer.writerow(res_row)
    res_eval_file.flush()

    pb.write("evaluating test...")
    ppl_test = eval_model(runner, data_pipeline(test_dataset, epochs=1, shuffle=False), len(test_dataset))

    res_row = {"epoch": epoch, "step": step, "dataset": "test", "perplexity": ppl_validation}
    res_eval_writer.writerow(res_row)
    res_eval_file.flush()

    pb.write("valid. ppl = {} \n test ppl {}".format(ppl_validation, ppl_test))

    return ppl_validation


# ======================================================================================
# TRAINING LOOP
# ======================================================================================
print("starting TF")

# preparing evaluation steps
# I use ceil because I make sure we have padded batches at the end
num_batches = np.ceil(len(training_dataset) / args.batch_size)
eval_step = np.ceil(len(training_dataset) / args.batch_size * args.eval_step)
epoch_step = 0
global_step = 0
current_epoch = 0
current_lr = args.learning_rate

last_eval = np.inf
current_eval = last_eval

model_runner.init_vars()
progress = tqdm(total=len(training_dataset) * args.epochs)
training_data = data_pipeline(training_dataset, epochs=args.epochs)
for ngram_batch in training_data:
    epoch = progress.n // len(training_dataset) + 1
    # ================================================
    # CHANGING EPOCH restart step
    # ================================================
    if epoch != current_epoch:
        current_epoch = epoch
        epoch_step = 0
        progress.write("epoch: {}".format(current_epoch))

    # ================================================
    # EVAL
    # ================================================
    if (epoch_step % eval_step) == 0:
        # write model state at the beginning of each eval epoch (not the first one)
        if args.model_eval_checkpoint and not global_step == 0 and epoch_step == 0:
            model_runner.save_model(model_name=model_path, step=global_step, write_state=False)

        current_eval = evaluation(model_runner, progress, epoch, global_step)

        # last eval is not defined
        if global_step == 0:
            last_eval = current_eval

        if args.lr_decay and (epoch_step == 0 or args.lr_decay_on_eval):
            if current_eval > last_eval:
                current_lr = current_lr * args.lr_decay_rate
                # only change last eval if we're updating weights on each eval
                progress.write("learning rate changed to {}".format(current_lr))

        last_eval = current_eval

    # ================================================
    # TRAIN MODEL
    # ================================================
    ngram_batch = np.array(ngram_batch, dtype=np.int64)
    ctx_ids = ngram_batch[:, :-1]
    word_ids = ngram_batch[:, -1]
    # one_hot = transform.batch_one_hot(word_ids, vocab_size)

    # model_runner.train(ctx_ids, one_hot, current_lr)
    model_runner.train(ctx_ids, word_ids, current_lr)
    progress.update(args.batch_size)

    epoch_step += 1
    global_step += 1

# write final evaluation at the end of the run
# write model state at the end of the run
evaluation(model_runner, progress, epoch, epoch_step)
model_runner.save_model(model_name=model_path, step=global_step, write_state=False)

model_runner.close_session()
progress.write("Processed {} n-grams".format(progress.n))
progress.close()

# close result files
res_eval_file.close()
