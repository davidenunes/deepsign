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
from deepsign.data.iterators import chunk_it, batch_it, shuffle_it, repeat_apply
from deepsign.models.lbl import LBL
from exp.args import ParamDict

default_corpus = os.path.join(os.getenv("HOME"), "data/datasets/ptb/")
default_out_dir = os.getcwd()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

defaults = {
    'id': (int, 0),
    'run': (int, 1),
    'run_id': (int, 0),  # run id
    'corpus': (str, default_corpus),
    'ngram_size': (int, 5),
    'save_model': (bool, False),
    'out_dir': (str, default_out_dir),
    # network architecture
    'embed_dim': (int, 128),
    'embed_init': (str, 'normal', ["normal", "uniform"]),
    'embed_init_val': (float, 0.01),
    'logit_init': (str, "uniform", ["normal", "uniform"]),
    'logit_init_val': (float, 0.01),
    'embed_share': (bool, True),
    'h_dim': (int, 256),
    'h_act': (str, "relu", ['relu', 'tanh', 'elu']),

    # lbl parameters
    'use_gate': (bool, True),
    'use_hidden': (bool, True),
    'embed_share': (bool, True),

    'x_to_f_init': (str, "uniform", ["normal", "uniform"]),
    'x_to_f_init_val': (float, 0.01),
    'h_to_f_init': (str, "uniform", ["normal", "uniform"]),
    'h_to_f_init_val': (float, 0.01),

    # training
    'epochs': (int, 2),
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
    'f_init': (str, "uniform", ["normal", "uniform"]),
    'f_init_val': (float, 0.01),

    # regularisation
    'clip_grads': (bool, True),
    # if true clips by local norm, else clip by norm of all gradients
    'clip_local': (bool, True),
    'clip_value': (float, 1.0),

    'dropout': (bool, True),
    'embed_dropout': (bool, True),
    'keep_prob': (float, 0.95),

    'l2_loss': (bool, False),
    'l2_loss_coef': (float, 1e-5),
}
arg_dict = ParamDict(defaults)


def run(**kwargs):
    arg_dict.from_dict(kwargs)
    args = arg_dict.to_namespace()

    # ======================================================================================
    # Load Params, Prepare results assets
    # ======================================================================================
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # print(args.corpus)

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
    # Load Corpus & Vocab
    # ======================================================================================
    corpus = h5py.File(os.path.join(args.corpus, "ptb_{}.hdf5".format(args.ngram_size)), mode='r')
    vocab = marisa_trie.Trie(corpus["vocabulary"])

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

    model = LBL(ctx_size=args.ngram_size - 1,
                vocab_size=len(vocab),
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
                l2_loss_coef=args.l2_loss_coef,
                use_nce=False)

    model_runner = tx.ModelRunner(model)

    # we use an InputParam because we might want to change it during training
    lr_param = tx.InputParam(value=args.lr)
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
        model_runner.config_optimizer(optimizer, optimizer_params=lr_param,
                                      gradient_op=clip_fn,
                                      global_gradient_op=not args.clip_local)
    else:
        model_runner.config_optimizer(optimizer, optimizer_params=lr_param)

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
        ppl_test = eval_model(runner, data_pipeline(test_data, epochs=1, shuffle=False), len(test_data),
                              display_progress)

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
    try:
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
    except Exception as e:
        traceback.print_exc()
        os.remove(ppl_file.name)
        os.remove(param_file.name)
        raise e


if __name__ == "__main__":
    # test with single gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    run()
