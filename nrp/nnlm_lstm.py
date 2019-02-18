import argparse
import csv
import marisa_trie
import os

import tensorflow as tf
import numpy as np
import h5py


from tqdm import tqdm
import traceback
import tensorx as tx
import tensorx.train
from deepsign.data.pipelines import to_parallel_seq
from exp.args import ParamDict
from deepsign.data.corpora.ptb import PTBReader
from tensorflow.python.ops.candidate_sampling_ops import uniform_candidate_sampler as uniform_sampler
from deepsign.data.corpora.wiki103 import WikiText103
from functools import partial

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class NNLM(tx.Model):
    """ Neural Probabilistic Language Model


    if use_f_predict is True, this model can be interpreted as an

    Energy-based Neural Network Language Modelling network

    Same as Bengio Neural Probabilistic Language Model but with a linear layer
    at the end feature_pred with the same dimensions as the embeddings and possibly
    with embedding sharing between input and output layers

    """

    def __init__(self,
                 inputs,
                 labels,
                 vocab_size,
                 embed_dim,
                 h_dim,
                 embed_init=tx.zeros_init(),
                 logit_init=tx.glorot_uniform(),
                 num_h=1,
                 h_activation=tx.tanh,
                 h_init=tx.glorot_uniform(),
                 w_dropconnect=None,
                 u_dropconnect=None,
                 r_dropout=0.4,
                 y_dropout=0.4,
                 embed_dropout=0.3,
                 other_dropout=0.3,
                 l2_loss=False,
                 l2_weight=1e-5,
                 use_f_predict=False,
                 f_init=tx.random_uniform(minval=-0.01, maxval=0.01),
                 embed_share=False,
                 logit_bias=False,
                 use_nce=False,
                 nce_samples=10,
                 skip_connections=False
                 ):
        if not isinstance(inputs, tx.Input):
            raise TypeError("inputs must be an Input layer")
        self.inputs = inputs
        self.labels = labels
        if not isinstance(labels, tx.Input):
            raise TypeError("labels must be an Input layer")

        if inputs.dtype != tf.int32 and inputs.dtype != tf.int64:
            raise TypeError("Invalid dtype for input: expected int32 or int64, got {}".format(inputs.dtype))

        if num_h < 0:
            raise ValueError("num hidden should be >= 0")

        # ===============================================
        # RUN GRAPH
        # ===============================================
        var_reg = []

        with tf.name_scope("run"):
            # feature lookup

            embeddings = tx.Lookup(inputs,
                                   seq_size=None,
                                   lookup_shape=[vocab_size, embed_dim],
                                   weight_init=embed_init)
            var_reg.append(embeddings.weights)
            feature_lookup = embeddings.permute_batch_time()

            last_layer = feature_lookup

            cell_proto = tx.LSTMCell.proto(
                n_units=h_dim,
                activation=h_activation,
                gate_activation=tx.hard_sigmoid,
                w_init=h_init,
                u_init=h_init,
                w_dropconnect=w_dropconnect,
                u_dropconnect=u_dropconnect,
                r_dropout=r_dropout,
                x_dropout=None,
                y_dropout=y_dropout,
                regularized=False,
                name="cell",
            )

            lstm_layers = []
            for i in range(num_h):
                lstm_layer = tx.RNN(last_layer,
                                    cell_proto=cell_proto,
                                    regularized=False,
                                    stateful=True,
                                    name="LSTM_{}".format(i + 1))

                lstm_layers.append(lstm_layer)

                var_reg += [wi.weights for wi in lstm_layer.cell.w]
                var_reg += [ui.weights for ui in lstm_layer.cell.u]

                last_layer = lstm_layer

            # last time step is the state used to make the prediction
            # last_layer = tx.Reshape(last_layer, [-1, h_dim])

            # TODO this is not consistent with locked dropout for the last layer
            # where the same mask should be applied across time steps
            # to do this I need either y_dropout to be available or some sort of map
            # operation I can use with layers outputting 3D tensors
            # something equivalent to https://keras.io/layers/wrappers/ which applies
            # a layer to every temporal slice of an input. They implement this the same way
            # they implement an RNN

            # feature prediction for Energy-Based Model
            if use_f_predict:
                last_layer = tx.Linear(last_layer, embed_dim, f_init, add_bias=True, name="f_predict")
                var_reg += last_layer.variables
                f_predict = last_layer

            shared_weights = feature_lookup.weights if embed_share else None
            transpose_weights = embed_share
            logit_init = logit_init if not embed_share else None
            run_logits = tx.Linear(last_layer,
                                   n_units=vocab_size,
                                   weight_init=logit_init,
                                   shared_weights=shared_weights,
                                   transpose_weights=transpose_weights,
                                   add_bias=logit_bias,
                                   name="logits")

            if not embed_share:
                var_reg.append(run_logits.weights)

            run_output = tx.Activation(run_logits, tx.softmax, name="run_output")

            # ===============================================
            # TRAIN GRAPH
            # ===============================================
            with tf.name_scope("train"):
                embeddings = embeddings.reuse_with(inputs)
                feature_lookup = embeddings.permute_batch_time()

                if embed_dropout:
                    feature_lookup = tx.Dropout(feature_lookup, probability=embed_dropout, name="drop_features")

                last_layer = feature_lookup

                for i in range(num_h):
                    lstm_layer = lstm_layers[i].reuse_with(last_layer, regularized=True)
                    last_layer = lstm_layer

                # last_layer = tx.Reshape(last_layer, [-1, h_dim])

                # feature prediction for Energy-Based Model
                if use_f_predict:
                    # last_layer = f_predict.reuse_with(last_layer)
                    last_layer = f_predict.reuse_with(last_layer)

                last_layer = tx.Dropout(last_layer, probability=other_dropout, locked=False)

                train_logits = run_logits.reuse_with(last_layer, name="train_logits")

                train_output = tx.Activation(train_logits, tx.softmax, name="run_output")

            def categorical_loss(labels, logits):
                # labels come as a batch of classes [[1,2],[3,4]] -> [1,3,2,4] time steps are ordered to match logits
                labels = tx.Transpose(labels)
                labels = tx.Reshape(labels, [-1])
                labels = tx.dense_one_hot(labels, num_cols=vocab_size)
                loss = tx.categorical_cross_entropy(labels=labels, logits=logits)

                return tf.reduce_mean(loss)

            def nce_loss(labels, weights, bias, predict):
                noise = uniform_sampler(labels, 1, nce_samples, True, vocab_size)
                loss = tf.nn.nce_loss(weights=weights,
                                      biases=bias,
                                      inputs=predict,
                                      labels=labels,
                                      num_sampled=nce_samples,
                                      num_classes=vocab_size,
                                      num_true=1,
                                      sampled_values=noise)
                return tf.reduce_mean(loss)

            if use_nce:
                bias = tx.VariableLayer(var_shape=[vocab_size], name="nce_bias")

                # wraps a layer to expose the weights as a layer but with the layer as its input
                nce_weights = tx.WrapLayer(embeddings,
                                           n_units=embeddings.n_units,
                                           wrap_fn=lambda x: x.weights,
                                           layer_fn=True)
                train_loss = tx.LambdaLayer(labels, nce_weights, bias, last_layer, apply_fn=nce_loss, name="nce_loss")
            else:
                train_loss = tx.LambdaLayer(labels, train_logits, apply_fn=categorical_loss,
                                            name="train_loss")

            if l2_loss:
                l2_losses = [tf.nn.l2_loss(var) for var in var_reg]
                train_loss = tx.LambdaLayer(train_loss,
                                            apply_fn=lambda x: x + l2_weight * tf.add_n(l2_losses),
                                            name="train_loss_l2")

        # ===============================================
        # EVAL GRAPH
        # ===============================================
        with tf.name_scope("eval"):
            eval_loss = tx.LambdaLayer(labels, run_logits, apply_fn=categorical_loss,
                                       name="eval_loss")

        self.stateful_layers = lstm_layers
        # BUILD MODEL
        super().__init__(run_outputs=run_output,
                         run_inputs=inputs,
                         train_inputs=[inputs, labels],
                         train_outputs=train_output,
                         train_loss=train_loss,
                         eval_inputs=[inputs, labels],
                         eval_outputs=run_output,
                         eval_score=eval_loss)

    def reset_state(self):
        """ returns an op that resets all stateful layers

        """
        reset_op = tf.group(*[layer.reset() for layer in self.stateful_layers])
        self.session.run(reset_op)





default_corpus = os.path.join(os.getenv("HOME"), "data/datasets/ptb/")
default_out_dir = os.getcwd()

defaults = {
    'pid': (int, 0),
    'id': (int, 0),  # param configuration id
    'run': (int, 1),  # run number within param id
    'corpus': (str, default_corpus),
    'seq_len': (int, 35),
    'min_seq_len': (int, 5),
    'seq_prob': (float, 0.95),
    'mark_eos': (bool, False),
    'save_model': (bool, False),
    'out_dir': (str, default_out_dir),
    # network architecture
    'embed_dim': (int, 400),
    'embed_init': (str, 'uniform', ["normal", "uniform", "zero"]),
    'embed_init_val': (float, 0.05),
    'logit_init': (str, "uniform", ["normal", "uniform"]),
    'logit_init_val': (float, 0.05),
    'h_init': (str, "xavier", ["normal", "uniform", "xavier", "he"]),
    'h_init_val': (float, 0.01),
    'embed_share': (bool, True),
    'num_h': (int, 1),
    'h_dim': (int, 650),
    'h_act': (str, "tanh", ['relu', 'tanh', 'elu', 'selu']),
    'epochs': (int, 100),

    # needed for adam and ams
    # training
    'batch_size': (int, 20),
    'optimizer': (str, 'sgd', ['sgd', 'adam']),
    'lr': (float, 1.0),
    'lr_decay': (bool, True),
    'lr_decay_rate': (float, 0.8),

    'optimizer_beta1': (float, 0.9),
    'optimizer_beta2': (float, 0.999),
    'optimizer_epsilon': (float, 1e-8),
    # annealing
    'lr_decay_threshold': (float, 1e-6),
    'eval_threshold': (float, 1.0),
    'early_stop': (bool, True),
    'patience': (int, 3),
    'use_f_predict': (bool, True),
    'f_init': (str, "uniform", ["normal", "uniform"]),
    'f_init_val': (float, 0.01),
    'logit_bias': (bool, False),

    # regularisation
    'clip_grads': (bool, True),
    'clip_local': (bool, False),
    'clip_value': (float, 1.0),

    'embed_dropout': (bool, True),
    'embed_drop_prob': (float, 0.4),

    'other_dropout': (bool, True),
    'other_drop_prob': (float, 0.4),

    'x_dropout': (bool, True),
    'x_drop_prob': (float, 0.4),

    'r_dropout': (bool, False),
    'r_drop_prob': (float, 0.4),

    'w_dropconnect': (bool, False),
    'w_drop_prob': (float, 0.1),

    'u_dropconnect': (bool, True),
    'u_drop_prob': (float, 0.5),

    'l2_loss': (bool, False),
    'l2_loss_coef': (float, 1e-5),

    "eval_test": (bool, True),
    "eval_train": (bool, False),

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

    to_parallel_batches = partial(to_parallel_seq,
                                  vocab=vocab,
                                  seq_len=args.seq_len,
                                  seq_prob=args.seq_prob,
                                  min_seq_len=args.min_seq_len,
                                  batch_size=args.batch_size,
                                  epochs=1,
                                  enum_seq=False,
                                  enum_epoch=False,
                                  return_future=True)

    # print("counting dataset samples...")
    training_len = sum(1 for _, _ in to_parallel_batches(corpus.training_set, batch_size=1))

    validation_len = None
    test_len = None
    if args.eval_progress:
        validation_len = sum(1 for _, _ in to_parallel_batches(corpus.validation_set, batch_size=1))
        test_len = sum(1 for _, _ in to_parallel_batches(corpus.test_set, batch_size=1))

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
        h_init = tx.random_normal(mean=0., stddev=args.h_init_val)
    elif args.h_init == "uniform":
        h_init = tx.random_uniform(minval=-args.h_init_val, maxval=args.h_init_val)
    elif args.h_init == "xavier":
        h_init = tx.glorot_uniform()
    elif args.h_init == "he":
        h_init = tx.he_normal_init()

    # Configure embedding and logit weight initializers
    if args.embed_init == "normal":
        embed_init = tx.random_normal(mean=0.,
                                      stddev=args.embed_init_val)
    elif args.embed_init == "uniform":
        embed_init = tx.random_uniform(minval=-args.embed_init_val,
                                       maxval=args.embed_init_val)
    elif args.embed_init == "zero":
        embed_init = tx.zeros_init()

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

    # dynamic inputs we don't know how many
    inputs = tx.Input(n_units=None, dtype=tf.int64, name="ctx_inputs")
    labels = tx.Input(n_units=None, dtype=tf.int64, name="ctx_labels")
    model = NNLM(inputs=inputs,
                 labels=labels,
                 vocab_size=len(vocab),
                 embed_dim=args.embed_dim,
                 embed_init=embed_init,
                 embed_share=args.embed_share,
                 skip_connections=False,
                 logit_init=logit_init,
                 h_dim=args.h_dim,
                 num_h=args.num_h,
                 h_activation=h_act,
                 h_init=h_init,
                 y_dropout=args.x_drop_prob if args.x_dropout else None,
                 r_dropout=args.r_drop_prob if args.r_dropout else None,
                 w_dropconnect=args.w_drop_prob if args.w_dropconnect else None,
                 u_dropconnect=args.u_drop_prob if args.u_dropconnect else None,
                 embed_dropout=args.embed_drop_prob if args.embed_dropout else None,
                 other_dropout=args.other_drop_prob if args.other_dropout else None,
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
    """
    lr_param = tensorx.train.StepDecay(value=args.lr,
                                       decay_after=6,
                                       decay_rate=args.lr_decay_rate,
                                       decay_threshold=args.lr_decay_threshold)

    """
    lr = lr_param.value
    # lr_param = tx.train.Param(value=args.lr)

    if args.optimizer == "sgd":
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr_param.tensor)
    elif args.optimizer == "adam":
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_param.tensor,
                                           beta1=args.optimizer_beta1,
                                           beta2=args.optimizer_beta2,
                                           epsilon=args.optimizer_epsilon)

    def clip_grad_global(grads):
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
        model.reset_state()
        if display_progress:
            pb = tqdm(total=len_dataset, ncols=60, position=1)
        batches_processed = 0
        sum_loss = 0
        for ctx, target in dataset_it:
            # batch = np.array(batch, dtype=np.int64)

            mean_loss = model.eval({inputs: ctx, labels: target})
            sum_loss += mean_loss

            if display_progress:
                pb.update(args.batch_size)
            batches_processed += 1

        if display_progress:
            pb.close()

        return np.exp(sum_loss / batches_processed)

    def evaluation(model: tx.Model, progress_bar, cur_epoch, step, display_progress=False):

        ppl_validation = eval_model(model,
                                    to_parallel_batches(corpus.validation_set),
                                    validation_len,
                                    display_progress)
        res_row = {"id": args.id, "run": args.run, "epoch": cur_epoch, "step": step, "lr": lr,
                   "dataset": "validation",
                   "perplexity": ppl_validation}
        ppl_writer.writerow(res_row)

        if args.eval_test:
            # pb.write("[Eval Test Set]")
            ppl_test = eval_model(model,
                                  to_parallel_batches(corpus.test_set),
                                  test_len,
                                  display_progress)

            res_row = {"id": args.id, "run": args.run, "epoch": cur_epoch, "step": step, "lr": lr,
                       "dataset": "test",
                       "perplexity": ppl_test}
            ppl_writer.writerow(res_row)

        if args.eval_train:
            # pb.write("[Eval Test Set]")
            ppl_test = eval_model(model,
                                  to_parallel_batches(corpus.training_set),
                                  training_len,
                                  display_progress)

            res_row = {"id": args.id, "run": args.run, "epoch": cur_epoch, "step": step, "lr": lr,
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
    current_seq = 0
    patience = 0

    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    sess = tf.Session(config=cfg)
    model.set_session(sess)
    model.init_vars()

    progress = tqdm(total=training_len * args.epochs, position=args.pid + 1, disable=not args.display_progress)
    training_data = to_parallel_batches(corpus.training_set,
                                        epochs=args.epochs,
                                        enum_epoch=True,
                                        enum_seq=True)

    eval_history = []
    num_improvs = 0

    try:

        for i, seq_batch in training_data:
            s, ctx, target = seq_batch
            epoch = i + 1

            # when we have multiple independent sequences withing an epoch
            if current_seq != s:
                model.reset_state()
                current_seq = s
            # Start New Epoch
            if epoch != current_epoch:
                current_epoch = epoch
                epoch_step = 0
                # reset model recurrent state
                model.reset_state()
                if args.display_progress:
                    progress.set_postfix({"epoch": current_epoch})

            # ================================================
            # EVALUATION
            # ================================================
            if epoch_step == 0:
                current_eval = evaluation(model, progress, epoch, global_step,
                                          display_progress=args.eval_progress)

                eval_history.append(current_eval)

                lr_param.value = lr
                lr_param.update(current_eval)
                lr = lr_param.value

                # EARLY STOP
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
            # seq_batch = np.array(seq_batch, dtype=np.int64)
            # ctx_ids = seq_batch[:, :-1]
            # word_ids = seq_batch[:, -1:]
            # reset sometimes to make it easier on eval time to start from 0
            # if np.random.random() < 0.01:
            #    model.reset_state()

            # set lr param based on linear rescaling so that we don't favour short sequences
            # seq_len = np.shape(ctx)[-1]
            # lr_param.value = seq_len / args.seq_len * lr

            try:
                model.train({inputs: ctx, labels: target})
            except Exception as e:
                raise e
            progress.update(args.batch_size)

            epoch_step += 1
            global_step += 1

        # if not early stop, evaluate last state of the model
        if not args.early_stop or patience < 3:
            current_eval = evaluation(model, progress, epoch, epoch_step)
            eval_history.append(current_eval)
        ppl_file.close()

        if args.save_model:
            model.save_model(model_name=model_path, step=global_step, write_state=False)

        model.close_session()
        progress.close()
        tf.reset_default_graph()

        # return the best validation evaluation
        return min(eval_history)

    except Exception as e:
        traceback.print_exc()
        os.remove(ppl_file.name)
        os.remove(param_file.name)
        raise e


if __name__ == "__main__":
    # test with single gpu 0 1 2 or 3 in my case
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    run(progress=True)
