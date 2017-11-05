from tensorx_old.layers import Input
from tensorx_old.models.nrp2 import NRP
from tensorx_old.init import glorot_init
from deepsign.rp.ri import Generator as RIGen
import numpy as np

import tensorflow as tf

# random index dimension
k = 100
s = 4
h_dim = 500
ri_gen = RIGen(active=s, dim=k)

r = ri_gen.generate()

labels_p = Input(n_units=k, name="ri_pos")
labels_n = Input(n_units=k, name="ri_neg")
labels = Input(n_units=k,name="ri_labels")

model = NRP(k_dim=k,h_dim=h_dim)
loss = model.get_loss(labels_p,labels_n)


optimizer = tf.train.AdagradOptimizer(0.1)
train_step = optimizer.minimize(loss)


# test model training
init = tf.global_variables_initializer()
with tf.Session() as ss:
    ss.run(init)

    x = np.asmatrix(r.to_vector())
    yp = x.copy()
    yp[yp < 0] = 0

    yn = x.copy()
    yn[yn > 0] = 0
    yn = np.abs(yn)

    print(x)
    print(yp)
    print(yn)

    for i in range(10000):
        ss.run(train_step, feed_dict={model.input(): x,
                                      labels_p(): yp,
                                      labels_n(): yn})

        if i % 100 == 0:
            result = ss.run(loss, feed_dict={model.input(): x,
                                             labels_p(): yp,
                                             labels_n(): yn})
            print(result)

    result = ss.run(model.output_sample(),feed_dict={model.input(): x})
    print(np.asmatrix(r.to_vector()))
    print(result)
