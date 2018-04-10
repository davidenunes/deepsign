import numpy as np
import unittest
import tensorflow as tf
import tensorx as tx
from deepsign.models.nnlm import NNLM
from deepsign.models.lbl import LBL
from deepsign.models.nrp import LBLNRP, RandomIndexTensor
from deepsign.rp.index import TrieSignIndex
from deepsign.rp.ri import Generator, RandomIndex
from deepsign.rp.tf_utils import to_sparse_tensor_value
import deepsign.data.views as views
import marisa_trie


class TestModels(unittest.TestCase):
    def setUp(self):
        self.ss = tf.InteractiveSession()

    def tearDown(self):
        self.ss.close()

    def test_lbl_nrp(self):
        vocab_size = 10000
        k = 10000
        s = 100

        generator = Generator(k, s)
        ris = [generator.generate() for _ in range(vocab_size)]
        ri_tensor = RandomIndexTensor.from_ri_list(ris, k, s)
        # ri_tensor = to_sparse_tensor_value(ris, k)

        model = LBLNRP(ctx_size=3,
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
        result = runner.run(np.array([[0, 2,1]]))
        print(np.shape(result))

        #runner.train(data=np.array([[0, 2, 1]]), loss_input_data=np.array([[1]]))

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


if __name__ == '__main__':
    unittest.main()
