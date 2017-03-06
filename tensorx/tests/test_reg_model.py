from tensorx.layers import Input
from tensorx.models.nrp import NRPRegression
from deepsign.rp.ri import Generator as RIGen
import numpy as np

import tensorflow as tf

# random index dimension
k = 1000
s = 10
h_dim = 300
ri_gen = RIGen(active=s, dim=k)

r = ri_gen.generate()

labels = Input(n_units=k, name="ri")

model = NRPRegression(k_dim=k,h_dim=h_dim)
loss = model.get_loss(labels)


optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.2)
train_step = optimizer.minimize(loss)


# test model training
init = tf.global_variables_initializer()
with tf.Session() as ss:
    ss.run(init)

    x = np.asmatrix(r.to_vector())


    for i in range(10000):
        ss.run(train_step, feed_dict={model.input(): x,
                                      labels(): x
                                      })

        if i % 100 == 0:
            result = ss.run(loss, feed_dict={model.input(): x,
                                             labels(): x
                                            })
            print(result)

    result = ss.run(model.output_sample(),feed_dict={model.input(): x})
    print(np.asmatrix(r.to_vector()))
    print(result)
