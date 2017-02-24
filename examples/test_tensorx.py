from tensorx.layers import Input, Dense, Act
from tensorx.init import glorot
from deepsign.rp.ri import Generator as RIGen
import numpy as np

import tensorflow as tf

# random index dimension
k = 1000
s = 10
h_dim = 300
ri_gen = RIGen(active=s, dim=k)

r = ri_gen.generate()
print(str(r))

pos_labels = Input(n_units=k, name="ri_pos")
neg_labels = Input(n_units=k, name="ri_neg")

labels = Input(n_units=k,name="ri_labels")

input = Input(n_units=k, name="ri")
h = Dense(input, n_units=h_dim, init=glorot, name="features")
pos_out = Dense(h, n_units=k, init=glorot, bias=True, name="pos_out")
neg_out = Dense(h, n_units=k, init=glorot, bias=True, name="neg_out")

#get positive or negative entries based on output probabilities
pos_sample = tf.cast(tf.less(tf.random_uniform([1,k]),tf.sigmoid(pos_out())),tf.float32)
neg_sample = tf.cast(tf.less(tf.random_uniform([1,k]),tf.sigmoid(neg_out())),tf.float32)
out = pos_sample-neg_sample


# 2 to the power of cross entropy of your language model with the test data
loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=pos_labels(), logits=pos_out()))
loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=neg_labels(), logits=neg_out()))

# 3 adding one weight regularisation
w_reg = tf.nn.l2_loss(pos_out.weights) * 0.001
w_reg += tf.nn.l2_loss(neg_out.weights) * 0.001

loss = ((loss1 + loss2) / 2.0) + w_reg
#loss = loss1 *0.5 + loss2 * 0.5

perplexity = tf.pow(2.0, loss)



#regression architecture
ri_out = Dense(h, n_units=k, init=glorot,act=Act.tanh, bias=True, name="ri_out")
#ri_loss = tf.



#optimizer = tf.train.AdagradOptimizer(0.1)
optimizer = tf.train.GradientDescentOptimizer(0.1)
train_step = optimizer.minimize(loss)

# Perplexity (PPL): Exponential of average negative log likelihood





init = tf.global_variables_initializer()
with tf.Session() as ss:
    ss.run(init)

    x = np.asmatrix(r.get_positive_vector())
    y_pos = np.asmatrix(r.get_positive_vector())
    y_neg = np.asmatrix(r.get_negative_vector())

    for i in range(10000):
        ss.run(train_step, feed_dict={input(): x,
                                      pos_labels(): y_pos,
                                      neg_labels(): y_neg
                                      })

        if i % 1000 == 0:
            result = ss.run(loss, feed_dict={input(): x,
                                                   pos_labels(): y_pos,
                                                   neg_labels(): y_neg
                                                   })
            print(result)


    result = ss.run(out,feed_dict={input(): x})
    print(r.to_vector())
    print(result)

# compile model
# sgd = SGD(lr=0.01, decay=1e-5)
# model.compile(optimizer=sgd,
#              loss='binary_crossentropy',
#              metrics=['binary_accuracy'])

# run one example


# x = to_tensor(r1.to_vector())
# y = to_tensor(r1.get_positive_vector())

# model.fit(x=x,y=y,nb_epoch=1,verbose=1,batch_size=1)


# y = model(x)
# print(K.eval(y))








# features = Dense(h_dim, init=xavier_init, name="features")(input)
# h = Activation("linear", name="h")(features)

# splits into two outputs
# pos_out = Dense(k, activation="sigmoid", init=xavier_init, name="pos_out")(h)
# neg_out = Dense(k, activation="sigmoid", init=xavier_init, name="neg_out")(h)

# model = Model(input=input, output=[pos_out,neg_out])




# we need to determine the appropriate lr
# sgd = SGD(lr=0.01, decay=1e-5)

# the loss is the sum of the two binary cross entropy
# model.compile(optimizer=sgd,
#              loss='binary_crossentropy',
#              metrics=['binary_accuracy'])





# test
# input
# x = ri_0.to_vector()
# x = np.asmatrix(x)

# output
# pos = np.zeros(k)
# neg = np.zeros(k)
# pos[ri_0.positive] = 1
# neg[ri_0.negative] = -1

# pos = np.asmatrix(pos)
# neg = np.asmatrix(neg)



# model.fit(np.transpose(np.asmatrix(x)), [pos, neg], verbose=1, nb_epoch=1)
