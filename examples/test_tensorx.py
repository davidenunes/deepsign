from tensorx.layers import Input
from tensorx.models.nrp import NRP
from tensorx.init import glorot
from deepsign.rp.ri import Generator as RIGen
import numpy as np

import tensorflow as tf

# random index dimension
k = 1000
s = 10
h_dim = 500
ri_gen = RIGen(active=s, dim=k)

r = ri_gen.generate()

labels_p = Input(n_units=k, name="ri_pos")
labels_n = Input(n_units=k, name="ri_neg")
labels = Input(n_units=k,name="ri_labels")

model = NRP(k_dim=k,h_dim=h_dim)
loss = model.get_loss(labels_p,labels_n)


optimizer = tf.train.GradientDescentOptimizer(0.1)
train_step = optimizer.minimize(loss)


# test model training
init = tf.global_variables_initializer()
with tf.Session() as ss:
    ss.run(init)

    x = np.asmatrix(r.get_positive_vector())
    y_pos = np.asmatrix(r.get_positive_vector())
    y_neg = np.asmatrix(r.get_negative_vector())

    for i in range(1000):
        ss.run(train_step, feed_dict={model.input(): x,
                                      labels_p(): y_pos,
                                      labels_n(): y_neg})

        if i % 100 == 0:
            result = ss.run(loss, feed_dict={model.input(): x,
                                             labels_p(): y_pos,
                                             labels_n(): y_neg})
            print(result)

    result = ss.run(model.output_sample(),feed_dict={model.input(): x})
    input()
    print(np.asmatrix(r.to_vector()))
    print(result)
