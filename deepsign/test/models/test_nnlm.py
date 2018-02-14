import unittest
import tensorflow as tf
import tensorx as tx
from deepsign.models.nnlm import NNLM


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.ss = tf.InteractiveSession()

    def tearDown(self):
        self.ss.close()

    def test_model_init(self):
        inputs = tx.TensorLayer([[1, 2]], 1, batch_size=1, dtype=tf.int32)
        loss_inputs = tx.TensorLayer([[1, 0, 0, 1]], 1, batch_size=1, dtype=tf.int32)

        model = NNLM(ctx_size=2,
                     vocab_size=4,
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


if __name__ == '__main__':
    unittest.main()
