import tensorflow as tf
import numpy as np
import tensorx as tx

from deepsign.rp import ri, tf_utils
from deepsign.rp.ri import Generator, RandomIndex
from deepsign.rp.tf_utils import to_sparse_tensor_value
from deepsign.data.views import chunk_it, batch_it, shuffle_it, repeat_fn, window_it
import marisa_trie
from deepsign.nlp.tokenization import Tokenizer
from deepsign.nlp import is_token

sentence = "mr anderson welcome back, we missed you."
tokenizer = Tokenizer()

tokens = tokenizer.tokenize(sentence)
tokens = [t for t in tokens if is_token.is_word(t)]
vocab = marisa_trie.Trie(tokens)

k = 10
s = 4
seq_size = 2
embed_dim = 4
batch_size = 2
generator = Generator(k, s)

print([vocab[w] for w in vocab.keys()])
ri_dict = {vocab[word]: generator.generate() for word in vocab.keys()}

tokens = [vocab[w] for w in tokens]
data_it = window_it(tokens, seq_size)
data_it = batch_it(data_it, batch_size)

vocab_tensor = [ri_dict[i] for i in range(len(vocab))]
sp_ri = tf_utils.to_sparse_tensor_value(vocab_tensor, dim=k)

inputs = tx.Input(n_units=2)
ri_inputs = tx.gather_sparse(sp_ri, inputs.tensor)
ri_inputs = tx.TensorLayer(ri_inputs, k)


embed = tx.Lookup(ri_inputs, seq_size, [k, embed_dim])

# logits: take the embeddings and get the features for all random indexes

ri_layer = tx.TensorLayer(sp_ri, n_units=k)
logits = tx.Linear(layer=ri_layer,
                   n_units=embed_dim,
                   shared_weights=embed.weights,
                   bias=True)


single_input = tx.Input(1)
ri_input = tx.TensorLayer(tx.gather_sparse(sp_ri,single_input.tensor),k)

logit = logits.reuse_with(ri_input)

session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

for batch in data_it:
    b = np.array(batch, dtype=np.int64)
    print("batch: \n", b)

    out = ri_inputs.tensor.eval({inputs.placeholder: b})

    ris = [ri_dict[i] for i in b.flatten()]

    print("len batch ris \n", len(ri_dict))
    print(out)

    for ri in ris:
        print(str(ri))

    print("logit shape\n", tf.shape(logits.tensor).eval())

    result = logits.tensor.eval()
    for i in range(len(result)):
        print('{} -> {}'.format(i, result[i]))

        g = logit.tensor.eval({single_input.placeholder: [[i]]})
        g = np.reshape(g,[-1])
        print('{} -> {}'.format(i, g))

