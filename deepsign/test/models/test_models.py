import numpy as np
import unittest
import tensorflow as tf
import tensorx as tx
from deepsign.models.nnlm import NNLM
from deepsign.models.lbl import LBL
from deepsign.models.nrp import LBL_NRP, RandomIndexTensor, NNLM_NRP
from deepsign.rp.index import TrieSignIndex
from deepsign.rp.ri import Generator, RandomIndex
from deepsign.rp.tf_utils import to_sparse_tensor_value
import deepsign.data.views as views
import marisa_trie
from deepsign.models.ri_nce import ri_nce_loss
from tqdm import tqdm
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TestModels(unittest.TestCase):
    def setUp(self):
        self.ss = tf.InteractiveSession()
        self.logdir = os.path.join(os.getenv("HOME"), "tmp")

    def tearDown(self):
        self.ss.close()

    def test_nnlm(self):
        vocab_size = 10000

        embed_dim = 512

        model = NNLM(ctx_size=3,
                     vocab_size=vocab_size,
                     embed_dim=embed_dim,
                     embed_share=False,
                     use_f_predict=True,
                     h_dim=500,
                     use_dropout=True,
                     embed_dropout=True
                     )
        runner = tx.ModelRunner(model)

        options = None
        runner.set_session(runtime_stats=True)
        runner.set_logdir(self.logdir)
        runner.log_graph()
        runner.config_optimizer(tf.train.GradientDescentOptimizer(learning_rate=0.05))

        for _ in tqdm(range(100)):
            # result = runner.train(np.array([[0, 2, 1]]),np.array([[0]]))
            result = runner.eval(np.array([[0, 2, 1]]), np.array([[0]]))

    def test_nnlm_nrp(self):
        vocab_size = 10000
        embed_dim = 128
        k = 5000
        s = 8

        generator = Generator(k, s)
        ris = [generator.generate() for _ in range(vocab_size)]
        ri_tensor = RandomIndexTensor.from_ri_list(ris, k, s)
        # ri_tensor = to_sparse_tensor_value(ris, k)
        # ri_tensor = tf.convert_to_tensor_or_sparse_tensor(ri_tensor)

        model = NNLM_NRP(ctx_size=3,
                         vocab_size=vocab_size,
                         k_dim=k,
                         s_active=s,
                         ri_tensor=ri_tensor,
                         embed_dim=embed_dim,
                         embed_share=True,
                         h_dim=100,
                         use_dropout=True,
                         embed_dropout=True
                         )
        runner = tx.ModelRunner(model)
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # options = None
        runner.set_session(runtime_stats=True, run_options=options)
        runner.set_logdir(self.logdir)
        runner.log_graph()
        runner.config_optimizer(tf.train.GradientDescentOptimizer(learning_rate=0.05))

        data = np.array([[0, 2, 1]])
        labels = np.array([[0]])

        for _ in tqdm(range(10)):
            runner.train(data, labels)
            # print(runner.eval(data, labels))
            # result = runner.eval(np.array([[0, 2, 1]]), np.array([[0]]))
            # result = runner.run(np.array([[0, 2, 1]]))

    def test_nce_nnlm_nrp(self):
        # ok for vocab bigger than 1000 with 10 samples, the eval keeps getting worse
        # gradient clipping too high also damages performance
        # lower dimensional k also turns the problem more challenging
        # so this depends more on k than on number of samples ?
        # also seems sensible to value of s, a bigger s makes the problem more difficult?
        # I know what it is: it's the fucking function that is malformed
        # I need to use tensorx lookup for the noise lookup otherwise I end up with a single vector
        # when using lookup sparse
        # when what I weant is multiple vectors, one for each noise sample
        vocab_size = 100
        k = 100
        s = 2
        embed_size = 100
        nce_samples = 2

        generator = Generator(k, s)
        ris = [generator.generate() for _ in range(vocab_size)]

        ri_tensor = to_sparse_tensor_value(ris, k)
        ri_tensor = tf.convert_to_tensor_or_sparse_tensor(ri_tensor)

        model = NNLM_NRP(ctx_size=3,
                         vocab_size=vocab_size,
                         k_dim=k,
                         num_h=2,
                         s_active=s,
                         h_activation=tx.relu,
                         ri_tensor=ri_tensor,
                         embed_dim=embed_size,
                         embed_share=True,
                         h_dim=256,
                         use_dropout=True,
                         embed_dropout=True,
                         keep_prob=0.75,
                         use_nce=True,
                         nce_samples=nce_samples,
                         )

        # model.eval_tensors.append(model.train_loss_tensors[0])
        runner = tx.ModelRunner(model)

        # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # options = None
        # runner.set_session(runtime_stats=True,run_options=options)

        runner.set_logdir("/tmp/")
        runner.log_graph()
        # runner.config_optimizer(tx.AMSGrad(learning_rate=0.00025))
        runner.config_optimizer(tf.train.RMSPropOptimizer(learning_rate=0.05)
                                #, gradient_op=lambda grad: tf.clip_by_norm(tf.Print(grad, [tf.reduce_mean(grad)]), 1.0))
                                ,gradient_op=lambda grad: tf.clip_by_norm(grad, 1.0))

        data = np.array([[0, 2, 1], [0, 2, 1]])
        labels = np.array([[1], [0]])
        # perplexity should be 2 on average

        for i in tqdm(range(3000)):
            res = runner.train(data, labels, output_loss=True)
            # print(res)
            if i % 30 == 0:
                #print(res)
                res = runner.eval(data, labels)
                print(res)
                #print("ppl ",np.exp(res))

    def test_nce_nnlm(self):
        vocab_size = 100
        k = 5000
        s = 2

        model = NNLM(ctx_size=3,
                     vocab_size=vocab_size,
                     h_activation=tx.relu,
                     embed_dim=64,
                     embed_share=True,
                     num_h=1,
                     h_dim=128,
                     use_f_predict=True,
                     use_dropout=True,
                     embed_dropout=True,
                     use_nce=True,
                     nce_samples=100
                     )

        # model.eval_tensors.append(model.train_loss_tensors[0])
        runner = tx.ModelRunner(model)

        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # options = None
        # runner.set_session(runtime_stats=True,run_options=options)
        runner.set_logdir("/tmp/")
        runner.log_graph()
        runner.config_optimizer(tf.train.GradientDescentOptimizer(learning_rate=0.4),
                                gradient_op=lambda grad: tf.clip_by_norm(grad, 1.0))

        data = np.array([[0, 1, 1]])
        labels = np.array([[2]])

        n = 1000
        for i in tqdm(range(n)):
            res = runner.train(data, labels, output_loss=True)
            # print(res)
            # print(res)
            #res = 0
            if i % 30 == 0:
                res = runner.eval(data, labels)
                print(res)
                #print("ppl ", np.exp(res / 30))
                #res = 0

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
        runner.set_logdir("/tmp/")
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

    def test_nrp_init(self):
        ri_dim = 1000

        inputs = tx.SparseInput(ri_dim, batch_size=1)
        loss_inputs = tx.Input(ri_dim * 2, batch_size=1)

        model = NRP(ctx_size=2,
                    ri_dim=ri_dim,
                    embed_dim=10,
                    batch_size=1,
                    h_dim=4,
                    num_h=2,
                    use_dropout=True,
                    keep_prob=0.1,
                    run_inputs=inputs,
                    loss_inputs=loss_inputs)

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

    def test_stage_nrp(self):
        ri_dim = 1000
        ri_s = 0.01
        batch_size = 2
        ngram_size = 3
        ctx_size = ngram_size - 1

        inputs = tx.SparseInput(ri_dim)
        loss_inputs = tx.Input(ri_dim * 2)

        # I was getting an invalid shape in the middle of running examples
        # hypothesis 1, LOOKUP doesnt work for sparse tensors , need to test it again
        # hypothesis 2, I messed up the parameters somewhere
        # CAUSE, BATCHES NEEDED TO BE PADDED, ... otherwise lookup doesnt work as it's supposed
        # I can add a dynamic batch param but
        # I could also pad AUTOMATICALLY
        # TODO add padding to lookup

        model = NRP(ctx_size=ctx_size,
                    ri_dim=ri_dim,
                    embed_dim=4,
                    batch_size=batch_size,
                    h_dim=3,
                    num_h=1,
                    use_dropout=False)

        runner = tx.ModelRunner(model)
        runner.config_optimizer(tf.train.GradientDescentOptimizer(0.05))
        runner.init_vars()

        # prepare dummy data
        samples = ["mr", "anderson", "welcome", "back", "we", "missed", "you"]
        samples_s = list(views.shuffle_it(samples, buffer_size=7))

        trie1 = marisa_trie.Trie(samples)
        trie2 = marisa_trie.Trie(samples_s)

        for w in samples:
            id1 = trie1[w]
            id2 = trie2[w]
            self.assertEqual(id1, id2)

        ri_gen = Generator(dim=ri_dim, num_active=ri_dim * ri_s)
        sign_index = TrieSignIndex(generator=ri_gen, vocabulary=samples, pregen_indexes=True)

        # s1: RandomIndex = sign_index.get_ri("mr")

        n_grams = views.window_it(samples, ngram_size)

        def to_id(i):
            return sign_index.get_id(i)

        def to_ri(i):
            s = sign_index.get_sign(i)
            return sign_index.get_ri(s)

        n_grams = [list(map(to_id, n)) for n in n_grams]

        n_grams = views.batch_it(n_grams, size=batch_size, padding=True, padding_elem=np.zeros([ngram_size]))
        # print("how data is usually fed \n", n_grams)
        for batch in n_grams:
            batch = np.array(batch)
            print("batch ", batch)

            ctx_ids = batch[:, :-1]

            ris = to_sparse_tensor_value(map(to_ri, ctx_ids.flatten()), ri_dim)

            target_ids = batch[:, -1:]

            # need to call flatten to make sure im iterating over the individual entries in map
            target_ris = map(to_ri, target_ids.flatten())
            target_ris = np.array([ri.to_class_vector() for ri in target_ris])
            # print("ctx: ", ris)
            # print("wid: ", target_ris)

            print(ris)
            print(target_ris)
            print(np.argwhere(target_ris == 1))

            runner.train(ris, target_ris)
            # res = runner.session.run(model.lookup.tensor,{model.run_inputs.placeholder: ris})
            # print(res)
            # res = runner.run(ris)

    def test_nce_ri(self):

        vocab_size = 4000
        k = 1000
        s = 4

        embed_dim = 100

        generator = Generator(k, s)
        ris = [generator.generate() for _ in range(vocab_size)]
        ri_tensor = RandomIndexTensor.from_ri_list(ris, k, s).to_sparse_tensor()

        inputs = tx.Input(n_units=2, dtype=tf.int32)
        labels = tx.Input(n_units=1, dtype=tf.int32)
        lookup = tx.Lookup(inputs, seq_size=2, feature_shape=[k, embed_dim])
        features = tx.Linear(lookup, n_units=embed_dim)

        nce_bias = tf.Variable(tf.zeros([k]), name='nce_bias', trainable=False)
        loss_base = tf.nn.nce_loss(lookup.weights, biases=nce_bias, labels=labels.tensor, inputs=features.tensor,
                                   num_sampled=1, num_classes=k, num_true=1)
        loss_base = tf.reduce_mean(loss_base)

        loss = ri_nce_loss(ri_tensor, weights=lookup.weights, inputs=features.tensor, labels=labels.tensor,
                           num_sampled=10,
                           num_classes=4000, num_true=1)
        loss = tf.reduce_mean(loss)

        learning = tf.train.GradientDescentOptimizer(learning_rate=0.005)  # momentum=0.8
        learning = learning.minimize(loss)

        tf.global_variables_initializer().run()

        data = np.array([[1, 3], [2, 3]])
        target = np.array([[0], [1]])
        feed = {inputs.placeholder: data, labels.placeholder: target}

        # print(loss_base.eval(feed))
        print(loss.eval(feed))

        for _ in range(5000):
            learning.run(feed)
            print(loss.eval(feed))


if __name__ == '__main__':
    unittest.main()
