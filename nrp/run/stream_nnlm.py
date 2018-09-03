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
from deepsign.data.views import chunk_it, take_it, batch_it, shuffle_it, repeat_it, repeat_apply, window_it, flatten_it
from deepsign.models.nnlm import NNLM
from exp.args import ParamDict
from deepsign.data.corpora.ptb import PTBReader
from deepsign.data.corpora.wiki103 import WikiText103

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

default_corpus = os.path.join(os.getenv("HOME"), "data/datasets/ptb/")
default_out_dir = os.getcwd()

defaults = {
    'run_id': (int, 0),  # run id
    'id': (int, 0),  # param id
    'run': (int, 1),  # run number within param id
    'corpus': (str, default_corpus),
    'ngram_size': (int, 5),
    'mark_eos': (bool, True),
    'save_model': (bool, False),
    'out_dir': (str, default_out_dir),
    # network architecture
    'embed_dim': (int, 128),
    'embed_init': (str, 'normal', ["normal", "uniform"]),
    'embed_init_val': (float, 0.01),
    'logit_init': (str, "uniform", ["normal", "uniform"]),
    'logit_init_val': (float, 0.01),
    'embed_share': (bool, True),
    'num_h': (int, 1),
    'h_dim': (int, 256),
    'h_act': (str, "relu", ['relu', 'tanh', 'elu']),
    'epochs': (int, 15),
    'batch_size': (int, 128),
    'shuffle': (bool, True),
    'shuffle_buffer_size': (int, 128 * 10000),
    'optimizer': (str, 'sgd', ['sgd', 'adam', 'ams']),
    # needed for adam and ams
    'optimizer_beta1': (float, 0.9),
    'optimizer_beta2': (float, 0.999),
    'optimizer_epsilon': (float, 1e-8),
    # training
    'lr': (float, 0.5),
    'lr_decay': (bool, True),
    'lr_decay_rate': (float, 0.5),
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
    # if true clips by local norm, else clip by norm of all gradients
    'clip_local': (bool, True),
    'clip_value': (float, 1.0),

    'dropout': (bool, True),
    'embed_dropout': (bool, True),
    'keep_prob': (float, 0.75),

    'l2_loss': (bool, False),
    'l2_loss_coef': (float, 1e-5),

    "eval_test": (bool, True)
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

    def corpus_pipeline(corpus_stream,
                        n_gram_size=args.ngram_size,
                        epochs=1,
                        batch_size=args.batch_size,
                        shuffle=args.shuffle,
                        flatten=False):
        """ Corpus Processing Pipeline.

        Transforms the corpus reader -a stream of sentences or words- into a stream of n-gram batches.

        Args:
            n_gram_size: the size of the n-gram window
            corpus_stream: the stream of sentences of words
            epochs: number of epochs we want to iterate over this corpus
            batch_size: batch size for the n-gram batch
            shuffle: if true, shuffles the n-grams according to a buffer size
            flatten: if true sliding windows are applied over a stream of words rather than within each sentence
            (n-grams can cross sentence boundaries)
        """

        if flatten:
            word_it = flatten_it(corpus_stream)
            n_grams = window_it(word_it, n_gram_size)
        else:
            sentence_n_grams = (window_it(sentence, n_gram_size) for sentence in corpus_stream)
            n_grams = flatten_it(sentence_n_grams)

        # at this point this is an n_gram iterator
        n_grams = ([vocab[w] for w in ngram] for ngram in n_grams)

        if epochs > 1:
            n_grams = repeat_it(n_grams, epochs)

        if shuffle:
            n_grams = shuffle_it(n_grams, args.shuffle_buffer_size)

        n_grams = batch_it(n_grams, size=batch_size, padding=False)
        return n_grams

    print("counting samples...")
    training_len = sum(1 for _ in corpus_pipeline(corpus.training_set(), batch_size=1, epochs=1, shuffle=False))
    validation_len = sum(1 for _ in corpus_pipeline(corpus.validation_set(), batch_size=1, epochs=1, shuffle=False))
    test_len = sum(1 for _ in corpus_pipeline(corpus.test_set(), batch_size=1, epochs=1, shuffle=False))

    # ======================================================================================
    # Load Params, Prepare results files
    # ======================================================================================
    # Experiment parameter summary
    res_param_filename = os.path.join(args.out_dir, "params_{id}.csv".format(id=args.run_id))
    with open(res_param_filename, "w") as param_file:
        writer = csv.DictWriter(f=param_file, fieldnames=arg_dict.keys())
        writer.writeheader()
        writer.writerow(arg_dict)
        param_file.flush()

    # make dir for model checkpoints
    if args.save_model:
        model_ckpt_dir = os.path.join(args.out_dir, "model_{id}".format(id=args.run_id))
        os.makedirs(model_ckpt_dir, exist_ok=True)
        model_path = os.path.join(model_ckpt_dir, "nnlm_{id}.ckpt".format(id=args.run_id))

    # start perplexity file
    ppl_header = ["id", "run", "epoch", "step", "lr", "dataset", "perplexity"]
    ppl_fname = os.path.join(args.out_dir, "perplexity_{id}.csv".format(id=args.run_id))

    ppl_file = open(ppl_fname, "w")
    ppl_writer = csv.DictWriter(f=ppl_file, fieldnames=ppl_header)
    ppl_writer.writeheader()

    # ======================================================================================
    # MODEL
    # ======================================================================================
    # Configure weight initializers based on activation functions
    if args.h_act == "relu":
        h_act = tx.relu
        h_init = tx.he_normal_init()
    elif args.h_act == "tanh":
        h_act = tx.tanh
        h_init = tx.xavier_init()
    elif args.h_act == "elu":
        h_act = tx.elu
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

    model = NNLM(ctx_size=args.ngram_size - 1,
                 vocab_size=len(vocab),
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
                 use_f_predict=args.use_f_predict,
                 f_init=f_init,
                 logit_bias=args.logit_bias)

    model_runner = tx.ModelRunner(model)

    # Input params can be changed during training by setting their value
    # lr_param = tx.InputParam(init_value=args.lr)
    lr_param = tx.EvalStepDecayParam(init_value=args.lr,
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

        ##pb.write("[Eval Validation Set]")

        ppl_validation = eval_model(runner, corpus_pipeline(corpus.validation_set(), epochs=1, shuffle=False),
                                    validation_len,
                                    display_progress)
        res_row = {"id": args.id, "run": args.run, "epoch": cur_epoch, "step": step, "lr": lr_param.value,
                   "dataset": "validation",
                   "perplexity": ppl_validation}
        ppl_writer.writerow(res_row)

        if args.eval_test:
            # pb.write("[Eval Test Set]")

            ppl_test = eval_model(runner, corpus_pipeline(corpus.test_set(), epochs=1, shuffle=False), test_len,
                                  display_progress)

            res_row = {"id": args.id, "run": args.run, "epoch": cur_epoch, "step": step, "lr": lr_param.value,
                       "dataset": "test",
                       "perplexity": ppl_test}
            ppl_writer.writerow(res_row)

        ppl_file.flush()

        if args.eval_test:
            pb.write("test. ppl = {}".format(ppl_test))

        # pb.write("valid. ppl = {}".format(ppl_validation))
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

    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    sess = tf.Session(config=cfg)
    model_runner.set_session(sess)
    model_runner.init_vars()

    print("dset len ", training_len)

    progress = tqdm(total=training_len * args.epochs, position=1)
    training_data = corpus_pipeline(corpus.training_set(),
                                    batch_size=args.batch_size,
                                    epochs=args.epochs,
                                    shuffle=args.shuffle)

    evals = []

    try:

        for ngram_batch in training_data:
            epoch = progress.n // training_len + 1
            # Start New Epoch
            if epoch != current_epoch:
                current_epoch = epoch
                epoch_step = 0
                if progress:
                    progress.write("epoch: {}".format(current_epoch))

            # Eval Time
            if epoch_step == 0:
                current_eval = evaluation(model_runner, progress, epoch, global_step)
                lr_param.update(current_eval)
                print(lr_param.eval_history)
                print("improvement ", lr_param.eval_improvement())

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
        tf.reset_default_graph()

    except Exception as e:
        traceback.print_exc()
        os.remove(ppl_file.name)
        os.remove(param_file.name)
        raise e


if __name__ == "__main__":
    # test with single gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    run(progress=True)
