import tensorx as tx
import tensorflow as tf
from deepsign.rp.ri import Generator
import numpy as np
from tqdm import tqdm
from deepsign.data.transform import ris_to_sp_tensor_value

ri_dim = 1000
ri_s = 10

gen = Generator(ri_dim, ri_s)

vocab_size = 10

ris = [gen.generate() for _ in range(vocab_size)]


dummy_logits = tf.constant(np.random.uniform(size=[10, ri_dim * 2]))
#out = tx.sigmoid(dummy_logits)
out = dummy_logits

with tf.Session() as ss:
    samples = tx.sample_sigmoid_from_logits(out, 100)
    #out_pos, out_neg = tf.split(samples, 2, axis=-1)

    # this seems slow
    for i in tqdm(range(1000)):
        #pos, neg = ss.run([out_pos, out_neg])
        s = ss.run(samples)

    #assert (np.shape(pos) == np.shape(neg))
    #print(np.shape(pos))
