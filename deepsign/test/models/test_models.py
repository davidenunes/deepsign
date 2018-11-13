import numpy as np
import unittest
import tensorflow as tf
import tensorx as tx
from deepsign.models.nnlm import NNLM
from deepsign.models.lbl import LBL
from deepsign.models.nrp import LBL_NRP, RandomIndexTensor, NNLM_NRP
from deepsign.models.nrp_nce import NRP
from deepsign.rp.ri import Generator, RandomIndex, ri_from_indexes
from deepsign.rp.index import TrieSignIndex
import deepsign.data.views as views
from deepsign.rp.tf_utils import ris_to_sp_tensor_value, generate_noise
from tqdm import tqdm
import os
from scipy.stats import chisquare, kstest, uniform
import time

import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="darkgrid")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TestModels(unittest.TestCase):
    def setUp(self):
        self.ss = tf.InteractiveSession()
        self.logdir = os.path.join(os.getenv("HOME"), "tmp")

    def tearDown(self):
        self.ss.close()

    def test_nnlm(self):
        vocab_size = 100000
        ctx_size = 20
        batch_size = 128
        embed_dim = 512

        model = NNLM(ctx_size=ctx_size,
                     vocab_size=vocab_size,
                     embed_dim=embed_dim,
                     embed_share=False,
                     use_f_predict=False,
                     h_dim=128,
                     use_dropout=True,
                     embed_dropout=True
                     )
        runner = tx.ModelRunner(model)

        options = None
        runner.set_session(runtime_stats=True)
        runner.set_log_dir(self.logdir)
        runner.log_graph()
        runner.config_optimizer(tf.train.GradientDescentOptimizer(learning_rate=0.05))

        data = np.random.randint(0, vocab_size, [batch_size, ctx_size])
        labels = np.random.randint(0, vocab_size, [batch_size, 1])

        for _ in tqdm(range(20)):
            # result = runner.train(np.array([[0, 2, 1]]),np.array([[0]]))
            result = runner.train(data, labels)

    def test_tx_random_choice(self):
        range_max = 100
        num_samples = 10
        batch_size = 10000

        samples = tx.choice(range_max, num_samples, batch_size)
        result = samples.eval()
        test = chisquare(result)
        plt.hist(result)
        plt.show()
        #print(test)



    def test_nnlm_nrp(self):
        vocab_size = 100000
        embed_dim = 512
        k = 4000
        s = 4
        ctx_size = 5
        batch_size = 128

        generator = Generator(k, s, symmetric=False)
        ris = [generator.generate() for _ in range(vocab_size)]
        ri_tensor = RandomIndexTensor.from_ri_list(ris, k, s)
        # ri_tensor = to_sparse_tensor_value(ris, k)
        # ri_tensor = tf.convert_to_tensor_or_sparse_tensor(ri_tensor)

        model = NNLM_NRP(ctx_size=ctx_size,
                         vocab_size=vocab_size,
                         k_dim=k,
                         s_active=s,
                         ri_tensor=ri_tensor,
                         embed_dim=embed_dim,
                         embed_share=False,
                         h_dim=128,
                         use_dropout=True,
                         embed_dropout=True
                         )
        runner = tx.ModelRunner(model)
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # options = None
        runner.set_session(runtime_stats=True, run_options=options)
        runner.set_log_dir(self.logdir)
        runner.log_graph()
        runner.config_optimizer(tf.train.GradientDescentOptimizer(learning_rate=0.05))

        data = np.random.randint(0, vocab_size, [batch_size, ctx_size])
        labels = np.random.randint(0, vocab_size, [batch_size, 1])

        for _ in tqdm(range(10)):
            runner.train(data, labels)

    def test_nce_nrp(self):
        vocab_size = 1000
        k = 500
        s = 8
        embed_size = 128
        nce_samples = 10
        noise_ratio = 0.1

        vocab = [str(i) for i in range(vocab_size)]

        generator = Generator(k, s)
        sign_index = TrieSignIndex(generator, vocabulary=vocab, pregen_indexes=True)
        ris = [sign_index.get_ri(sign_index.get_sign(i)) for i in range(len(sign_index))]
        # ris = [generator.generate() for _ in range(vocab_size)]

        ri_tensor = ris_to_sp_tensor_value(ri_seq=ris,
                                           dim=k,
                                           all_positive=False)

        ri_tensor_input = tx.SparseInput(n_units=k, value=ri_tensor)

        model = NRP(ctx_size=2,
                    vocab_size=vocab_size,
                    k_dim=k,
                    ri_tensor_input=ri_tensor_input,  # current dictionary state
                    embed_dim=embed_size,
                    h_dim=128,
                    num_h=1,
                    h_activation=tx.relu,
                    use_dropout=True,
                    embed_dropout=True,
                    keep_prob=0.70,
                    nce_samples=nce_samples,
                    nce_noise_amount=noise_ratio
                    )

        tf.summary.histogram("embeddings", model.embeddings.weights)
        for h in model.h_layers:
            tf.summary.histogram("h", h.linear.weights)

        # model.eval_tensors.append(model.train_loss_tensors[0])
        runner = tx.ModelRunner(model)
        runner.set_log_dir("/tmp")
        runner.log_graph()
        print("log graph")

        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # options = None
        runner.set_session(runtime_stats=True, run_options=options)

        # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

        # runner.config_optimizer(tf.train.GradientDescentOptimizer(learning_rate=0.005))#,
        # SGD with 0.025

        # lr = tx.InputParam(init_value=0.0002)
        lr = tx.InputParam(init_value=0.025)
        # runner.config_optimizer(tf.train.AdamOptimizer(learning_rate=lr.tensor, beta1=0.9), params=lr,
        runner.config_optimizer(tf.train.GradientDescentOptimizer(learning_rate=lr.tensor), params=lr,
                                global_gradient_op=False,
                                # gradient_op=lambda grad: tf.clip_by_global_norm(grad, 10.0)[0])
                                gradient_op=lambda grad: tf.clip_by_norm(grad, 1.0))

        data = np.array([[0, 2], [5, 7], [9, 8], [3, 4], [1, 9], [12, 8]])
        labels = np.array([[32], [56], [12], [2], [5], [23]])

        ppl_curve = []
        n = 256
        batch_size = 128

        dataset = np.column_stack((data, labels))
        # print(dataset)
        dataset = views.repeat_it([dataset], n)
        dataset = views.flatten_it(dataset)
        # shuffle 5 at a time
        dataset = views.shuffle_it(dataset, 6)
        dataset = views.batch_it(dataset, batch_size)

        # print(np.array(list(dataset)))
        # d = list(views.take_it(1, views.shuffle_it(d, 4)))[0]

        data_stream = dataset

        i = 0
        for data_stream in tqdm(data_stream, total=n * 5 / batch_size):
            sample = np.array(data_stream)

            t0 = time.time()
            ctx = sample[:, :-1]
            ctx.flatten()
            ctx = ctx.flatten()
            ctx_ris = [sign_index.get_ri(sign_index.get_sign(i)) for i in ctx]
            ctx_ris = ris_to_sp_tensor_value(ctx_ris,
                                             dim=sign_index.feature_dim(),
                                             all_positive=not sign_index.generator.symmetric)
            lbl_ids = sample[:, -1:]
            lbl = lbl_ids.flatten()
            lbl_ris = [sign_index.get_ri(sign_index.get_sign(i)) for i in lbl]
            lbl_ris = ris_to_sp_tensor_value(lbl_ris,
                                             dim=sign_index.feature_dim(),
                                             all_positive=not sign_index.generator.symmetric)

            noise = generate_noise(k_dim=k,
                                   batch_size=lbl_ris.dense_shape[0] * nce_samples,
                                   ratio=noise_ratio)
            t1 = time.time()
            # print(t1 - t0)
            # tf.summary.scalar("ctx convert time", t1 - t0)

            runner.train(ctx_ris, [lbl_ris, noise], output_loss=True, write_summaries=True)

        runner.close_session()

    #            if i % 5 == 0:
    #                res = runner.eval(ctx_ris, lbl_ids, write_summaries=False)
    #                if np.isnan(res):
    #                    print()
    #                ppl_curve.append(np.exp(res))
    #            i += 1
    #        print(i)

    #        ppl = sns.lineplot(x=np.array(list(range(len(ppl_curve)))), y=np.array(ppl_curve))
    #        print(ppl_curve)
    #        plt.show()
    # print("ppl ",np.exp(res))

    def test_nce_nnlm(self):
        vocab_size = 1000
        embed_size = 100
        nce_samples = 10

        model = NNLM(ctx_size=2,
                     vocab_size=vocab_size,
                     h_activation=tx.relu,
                     embed_dim=embed_size,
                     embed_share=True,
                     num_h=1,
                     h_dim=128,
                     use_f_predict=True,
                     use_dropout=True,
                     keep_prob=0.75,
                     embed_dropout=True,
                     use_nce=False,
                     nce_samples=nce_samples
                     )

        # model.eval_tensors.append(model.train_loss_tensors[0])
        runner = tx.ModelRunner(model)

        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # options = None
        runner.set_session(runtime_stats=True, run_options=options)
        runner.set_log_dir("/tmp/")
        runner.log_graph()
        runner.config_optimizer(tf.train.GradientDescentOptimizer(learning_rate=0.01),
                                gradient_op=lambda grad: tf.clip_by_norm(grad, 1.0))
        # runner.config_optimizer(tf.train.AdamOptimizer(learning_rate=0.005))

        data = np.array([[0, 2], [5, 7], [9, 8], [3, 4], [1, 9]])
        labels = np.array([[32], [56], [12], [2], [5]])
        # data = np.array([[0, 2], [5, 7], [9, 8], [3, 4], [3, 2]])
        # labels = np.array([[32], [56], [12], [2], [7]])
        # data = np.array([[0, 2]])
        # labels = np.array([[32]])

        ppl_curve = []

        for i in tqdm(range(3000)):
            res = runner.train(data, labels, output_loss=True)

            # print(res)
            # print(res)
            if i % 5 == 0:
                # print(res)
                res = runner.eval(data, labels)
                print(res)
                ppl_curve.append(np.exp(res))

        ppl = sns.lineplot(x=np.array(list(range(len(ppl_curve)))), y=np.array(ppl_curve))
        print(ppl_curve)
        plt.show()

    def test_lbl_nrp(self):
        vocab_size = 10000
        k = 10000
        s = 1000

        generator = Generator(k, s)
        ris = [generator.generate() for _ in range(vocab_size)]
        ri_tensor = RandomIndexTensor.from_ri_list(ris, k, s)
        # ri_tensor = to_sparse_tensor_value(ris, k)

        model = LBL_NRP(ctx_size=3,
                        vocab_size=vocab_size,
                        k_dim=k,
                        ri_tensor=ri_tensor,
                        embed_dim=10,
                        embed_share=True,
                        use_gate=True,
                        use_hidden=True,
                        h_dim=4,
                        use_dropout=True,
                        embed_dropout=True
                        )

        runner = tx.ModelRunner(model)
        # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        options = None
        runner.set_session(runtime_stats=True, run_options=options)
        runner.set_log_dir("/tmp/")
        runner.log_graph()
        runner.config_optimizer(tf.train.GradientDescentOptimizer(learning_rate=0.05))
        result = runner.run(np.array([[0, 2, 1]]))
        print(np.shape(result))

        # runner.train(data=np.array([[0, 2, 1]]), loss_input_data=np.array([[1]]))

    def test_lbl(self):
        model = LBL(ctx_size=2,
                    vocab_size=10000,
                    embed_dim=10,
                    embed_share=True,
                    use_gate=True,
                    use_hidden=True,
                    h_dim=4,
                    use_dropout=True,
                    embed_dropout=True)

        print("RUN GRAPH:")
        for layer in tx.layers_to_list(model.run_out_layers):
            print(layer.full_str())

        print("=" * 60)

        runner = tx.ModelRunner(model)
        runner.set_session(runtime_stats=True)
        runner.log_graph("/tmp/")
        runner.run([[0, 2]])

    def test_baseline_nnlm_init(self):
        model = NNLM(ctx_size=2,
                     vocab_size=4,
                     embed_dim=10,
                     h_dim=4,
                     num_h=2,
                     use_dropout=True,
                     embed_dropout=True,
                     keep_prob=0.1)

        print("RUN GRAPH:")
        for layer in tx.layers_to_list(model.run_out_layers):
            print(layer.full_str())

        print("=" * 60)

        print("TRAIN GRAPH:")
        for layer in tx.layers_to_list(model.train_out_layers):
            print(layer.full_str())
        print("=" * 60)

        print("EVAL GRAPH:")
        for layer in tx.layers_to_list(model.eval_out_layers):
            print(layer.full_str())
        print("=" * 60)

        runner = tx.ModelRunner(model)
        runner.log_graph("/tmp/")


if __name__ == '__main__':
    unittest.main()
