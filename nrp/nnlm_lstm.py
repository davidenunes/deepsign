import argparse
import csv
import marisa_trie
import os

import h5py
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import traceback
import tensorx as tx
import tensorx.train
from deepsign.data.pipelines import to_parallel_seq
from deepsign.models.nnlm_lstm import LSTM_NNLM
from exp.args import ParamDict
from deepsign.data.corpora.ptb import PTBReader
from deepsign.data.corpora.wiki103 import WikiText103
from functools import partial

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

default_corpus = os.path.join(os.getenv("HOME"), "data/datasets/ptb/")
default_out_dir = os.getcwd()

defaults = {
    'pid': (int, 0),
    'id': (int, 0),  # param configuration id
    'run': (int, 1),  # run number within param id
    'corpus': (str, default_corpus),
    'ngram_size': (int, 20),
    'mark_eos': (bool, False),
    'save_model': (bool, False),
    'out_dir': (str, default_out_dir),
    # network architecture
    'embed_dim': (int, 400),
    'embed_init': (str, 'uniform', ["normal", "uniform"]),
    'embed_init_val': (float, 0.05),
    'logit_init': (str, "uniform", ["normal", "uniform"]),
    'logit_init_val': (float, 0.05),
    'h_init': (str, "uniform", ["normal", "uniform", "xavier", "he"]),
    'h_init_val': (float, 0.05),
    'embed_share': (bool, True),
    'num_h': (int, 1),
    'h_dim': (int, 650),
    # TODO selo return overflow
    'h_act': (str, "tanh", ['relu', 'tanh', 'elu', 'selu']),
    'epochs': (int, 20),
    'batch_size': (int, 128),
    'shuffle': (bool, False),
    'shuffle_buffer_size': (int, 128 * 10000),
    # TODO ams produced an error
    'optimizer': (str, 'adagrad', ['sgd', 'adam', 'ams', 'adagrad']),
    # needed for adam and ams
    'optimizer_beta1': (float, 0.9),
    'optimizer_beta2': (float, 0.999),
    'optimizer_epsilon': (float, 1e-8),
    # training
    'lr': (float, 0.5),
    'lr_decay': (bool, True),
    'lr_decay_rate': (float, 0.8),
    # annealing
    'lr_decay_threshold': (float, 1e-6),
    # lr decay when last_ppl - current_ppl < eval_threshold
    'eval_threshold': (float, 1.0),
    'early_stop': (bool, True),
    'patience': (int, 3),
    'use_f_predict': (bool, True),
    'f_init': (str, "uniform", ["normal", "uniform"]),
    'f_init_val': (float, 0.01),
    'logit_bias': (bool, False),

    # regularisation
    'clip_grads': (bool, True),
    'clip_local': (bool, True),
    'clip_value': (float, 1.0),

    'embed_dropout': (bool, True),
    'embed_keep_prob': (float, 0.75),

    'w_dropout': (bool, True),
    'w_keep_prob': (float, 0.50),

    'u_dropconnect': (bool, False),
    'u_keep_prob': (float, 0.60),

    'reset_state': (bool, True),

    'l2_loss': (bool, True),
    'l2_loss_coef': (float, 1e-5),

    "eval_test": (bool, False),

    "eval_train": (bool, True),

    # eval display progress:
    "eval_progress": (bool, True),
    "display_progress": (bool, True)
}
arg_dict = ParamDict(defaults)


def run(**kwargs):
    arg_dict.from_dict(kwargs)
    args = arg_dict.to_namespace()

    # ======================================================================================
    # Load Corpus & Vocab
    # ======================================================================================
    corpus = PTBReader(path=args.corpus, mark_eos=args.mark_eos)
    corpus_stats = h5py.File(os.path.join(args.corpus, "ptb_stats.hdf5"), mode='r')
    vocab = marisa_trie.Trie(corpus_stats["vocabulary"])

    to_ngrams_batch = partial(to_parallel_seq,
                              vocab=vocab,
                              ngram_size=args.ngram_size,
                              batch_size=args.batch_size,
                              epochs=1,
                              shuffle=False,
                              shuffle_buffer_size=args.shuffle_buffer_size,
                              enum_epoch=False)

    # print("counting dataset samples...")
    training_len = sum(1 for _ in corpus_pipeline(corpus.training_set(), batch_size=1, epochs=1, shuffle=False))
    validation_len = None
    test_len = None
    if args.eval_progress:
        validation_len = sum(1 for _ in corpus_pipeline(corpus.validation_set(), batch_size=1, epochs=1, shuffle=False))
        test_len = sum(1 for _ in corpus_pipeline(corpus.test_set(), batch_size=1, epochs=1, shuffle=False))
    # print("done")
    # print("dset len ", training_len)

    # ======================================================================================
    # Load Params, Prepare results assets
    # ======================================================================================
    # Experiment parameter summary
    res_param_filename = os.path.join(args.out_dir, "params_{id}_{run}.csv".format(id=args.id, run=args.run))
    with open(res_param_filename, "w") as param_file:
        writer = csv.DictWriter(f=param_file, fieldnames=arg_dict.keys())
        writer.writeheader()
        writer.writerow(arg_dict)
        param_file.flush()

    # make dir for model checkpoints
    if args.save_model:
        model_ckpt_dir = os.path.join(args.out_dir, "model_{id}_{run}".format(id=args.id, run=args.run))
        os.makedirs(model_ckpt_dir, exist_ok=True)
        model_path = os.path.join(model_ckpt_dir, "nnlm_{id}_{run}.ckpt".format(id=args.id, run=args.run))

    # start perplexity file
    ppl_header = ["id", "run", "epoch", "step", "lr", "dataset", "perplexity"]
    ppl_fname = os.path.join(args.out_dir, "perplexity_{id}_{run}.csv".format(id=args.id, run=args.run))

    ppl_file = open(ppl_fname, "w")
    ppl_writer = csv.DictWriter(f=ppl_file, fieldnames=ppl_header)
    ppl_writer.writeheader()

    # ======================================================================================
    # MODEL
    # ======================================================================================
    # Configure weight initializers based on activation functions
    if args.h_act == "relu":
        h_act = tx.relu
    elif args.h_act == "tanh":
        h_act = tx.tanh
    elif args.h_act == "elu":
        h_act = tx.elu
    elif args.h_act == "selu":
        h_act = tf.nn.selu

    if args.h_init == "normal":
        h_init = tx.random_normal(mean=0.,
                                  stddev=args.h_init_val)
    elif args.h_init == "uniform":
        h_init = tx.random_uniform(minval=-args.h_init_val,
                                   maxval=args.h_init_val)
    elif args.h_init == "xavier":
        h_init = tx.xavier_init()
    elif args.h_init == "he":
        h_init = tx.he_normal_init()

    # Configure embedding and logit weight initializers
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

    f_init = None
    if args.use_f_predict:
        if args.f_init == "normal":
            f_init = tx.random_normal(mean=0., stddev=args.f_init_val)
        elif args.f_init == "uniform":
            f_init = tx.random_uniform(minval=-args.f_init_val, maxval=args.f_init_val)

    inputs = tx.Input(args.ngram_size - 1, dtype=tf.int64, name="ctx_inputs")
    labels = tx.Input(1, dtype=tf.int64, name="ctx_inputs")
    model = LSTM_NNLM(inputs=inputs,
                      labels=labels,
                      vocab_size=len(vocab),
                      embed_dim=args.embed_dim,
                      embed_init=embed_init,
                      embed_share=args.embed_share,
                      logit_init=logit_init,
                      h_dim=args.h_dim,
                      num_h=args.num_h,
                      h_activation=h_act,
                      h_init=h_init,
                      reset_state=args.reset_state,
                      w_dropout=args.w_dropout,
                      u_dropconnect=args.u_dropconnect,
                      w_keep_prob=args.w_keep_prob,
                      u_keep_prob=args.u_keep_prob,
                      embed_keep_prob=args.embed_keep_prob,
                      embed_dropout=args.embed_dropout,
                      l2_loss=args.l2_loss,
                      l2_weight=args.l2_loss_coef,
                      use_f_predict=args.use_f_predict,
                      f_init=f_init,
                      logit_bias=args.logit_bias,
                      use_nce=False)

    # Input params can be changed during training by setting their value
    # lr_param = tx.InputParam(init_value=args.lr)
    lr_param = tensorx.train.EvalStepDecayParam(value=args.lr,
                                                improvement_threshold=args.eval_threshold,
                                                less_is_better=True,
                                                decay_rate=args.lr_decay_rate,
                                                decay_threshold=args.lr_decay_threshold)
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
    elif args.optimizer == "adagrad":
        optimizer = tf.train.AdagradOptimizer(learning_rate=lr_param.tensor,)

    def clip_grad_global(grads):
        #TODO found hard coded value
        grads, _ = tf.clip_by_global_norm(grads, args.clip_value)
        return grads

    def clip_grad_local(grad):
        return tf.clip_by_norm(grad, args.clip_value)

    if args.clip_grads:
        if args.clip_local:
            clip_fn = clip_grad_local
        else:
            clip_fn = clip_grad_global

    if args.clip_grads:
        model.config_optimizer(optimizer, optimizer_params=lr_param,
                               gradient_op=clip_fn,
                               global_gradient_op=not args.clip_local)
    else:
        model.config_optimizer(optimizer, optimizer_params=lr_param)

    # ======================================================================================
    # EVALUATION
    # ======================================================================================

    def eval_model(model, dataset_it, len_dataset=None, display_progress=False):
        if display_progress:
            pb = tqdm(total=len_dataset, ncols=60, position=1)
        batches_processed = 0
        sum_loss = 0
        for batch in dataset_it:
            batch = np.array(batch, dtype=np.int64)
            ctx = batch[:, :-1]
            target = batch[:, -1:]

            mean_loss = model.eval({inputs: ctx, labels: target})
            sum_loss += mean_loss

            if display_progress:
                pb.update(args.batch_size)
            batches_processed += 1

        if display_progress:
            pb.close()

        return np.exp(sum_loss / batches_processed)

    def evaluation(model: tx.Model, progress_bar, cur_epoch, step, display_progress=False):

        ppl_validation = eval_model(model, corpus_pipeline(corpus.validation_set(), epochs=1, shuffle=False),
                                    validation_len,
                                    display_progress)
        res_row = {"id": args.id, "run": args.run, "epoch": cur_epoch, "step": step, "lr": lr_param.value,
                   "dataset": "validation",
                   "perplexity": ppl_validation}
        ppl_writer.writerow(res_row)

        if args.eval_test:
            # pb.write("[Eval Test Set]")
            ppl_test = eval_model(model, corpus_pipeline(corpus.test_set(), epochs=1, shuffle=False), test_len,
                                  display_progress)

            res_row = {"id": args.id, "run": args.run, "epoch": cur_epoch, "step": step, "lr": lr_param.value,
                       "dataset": "test",
                       "perplexity": ppl_test}
            ppl_writer.writerow(res_row)

        if args.eval_train:
            # pb.write("[Eval Test Set]")
            ppl_test = eval_model(model, corpus_pipeline(corpus.training_set(), epochs=1, shuffle=False), training_len,
                                  display_progress)

            res_row = {"id": args.id, "run": args.run, "epoch": cur_epoch, "step": step, "lr": lr_param.value,
                       "dataset": "train",
                       "perplexity": ppl_test}
            ppl_writer.writerow(res_row)

        ppl_file.flush()

        if args.eval_test:
            progress_bar.set_postfix({"test PPL ": ppl_test})

        # pb.write("valid. ppl = {}".format(ppl_validation))
        return ppl_validation

    # ======================================================================================
    # TRAINING LOOP
    # ======================================================================================
    # print("Starting TensorFlow Session")

    # preparing evaluation steps
    # I use ceil because I make sure we have padded batches at the end

    epoch_step = 0
    global_step = 0
    current_epoch = 0
    patience = 0

    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    sess = tf.Session(config=cfg)
    model.set_session(sess)
    model.init_vars()

    progress = tqdm(total=training_len * args.epochs, position=args.pid + 1, disable=not args.display_progress)
    training_data = corpus_pipeline(corpus.training_set(),
                                    batch_size=args.batch_size,
                                    epochs=args.epochs,
                                    shuffle=args.shuffle)

    evaluations = []

    try:

        for ngram_batch in training_data:
            epoch = progress.n // training_len + 1
            # Start New Epoch
            if epoch != current_epoch:
                current_epoch = epoch
                epoch_step = 0
                if args.display_progress:
                    progress.set_postfix({"epoch": current_epoch})

            # ================================================
            # EVALUATION
            # ================================================
            if epoch_step == 0:
                current_eval = evaluation(model, progress, epoch, global_step,
                                          display_progress=args.eval_progress)

                evaluations.append(current_eval)
                lr_param.update(current_eval)
                # print(lr_param.eval_history)
                # print("improvement ", lr_param.eval_improvement())

                if global_step > 0:
                    if args.early_stop and epoch > 1:
                        if lr_param.eval_improvement() < lr_param.improvement_threshold:
                            if patience >= 3:
                                break
                            patience += 1
                        else:
                            patience = 0

            # ================================================
            # TRAIN MODEL
            # ================================================
            ngram_batch = np.array(ngram_batch, dtype=np.int64)
            ctx_ids = ngram_batch[:, :-1]
            word_ids = ngram_batch[:, -1:]

            model.train({inputs: ctx_ids, labels: word_ids})
            progress.update(args.batch_size)

            epoch_step += 1
            global_step += 1

        # if not early stop, evaluate last state of the model
        if not args.early_stop or patience < 3:
            current_eval = evaluation(model, progress, epoch, epoch_step)
            evaluations.append(current_eval)
        ppl_file.close()

        if args.save_model:
            model.save_model(model_name=model_path, step=global_step, write_state=False)

        model.close_session()
        progress.close()
        tf.reset_default_graph()

        # return the best validation evaluation
        return min(evaluations)

    except Exception as e:
        traceback.print_exc()
        os.remove(ppl_file.name)
        os.remove(param_file.name)
        raise e


if __name__ == "__main__":
    # test with single gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    run(progress=True)
