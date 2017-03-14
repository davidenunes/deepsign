from tensorx.layers import Input
from tensorx.models.nrp import NRPCBow
from deepsign.rp.ri import Generator as RIGen
import numpy as np

import tensorflow as tf

# random index dimension
k = 1000
s = 10
h_dim = 2
ri_gen = RIGen(active=s, dim=k)

r1 = ri_gen.generate()
r2 = ri_gen.generate()

labels = Input(n_units=k*2, name="ri")

model = NRPCBow(k_dim=k,h_dim=h_dim)
loss = model.get_loss(labels)


optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.5)
train_step = optimizer.minimize(loss)


# test model training
init = tf.global_variables_initializer()
with tf.Session() as ss:
    ss.run(init)

    x = np.asmatrix([r1.to_vector(),r2.to_vector()])
    #print(x)

    y = np.asmatrix([r1.to_dist_vector(),r2.to_dist_vector()])
    #y = np.asmatrix(r.to_class_vector())

    for i in range(1000):
        ss.run(train_step, feed_dict={model.input(): x,
                                      labels(): y
                                      })

        if i % 100 == 0:
            result = ss.run(loss, feed_dict={model.input(): x,
                                             labels(): y
                                            })
            print(result)

    result = ss.run(model.output_sample(),feed_dict={model.input(): x})
    print("expected: ", x[0])
    #print("expected y: ",y)
    print("predicted: ",result[0])
