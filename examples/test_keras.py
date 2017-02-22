from keras.backend.tensorflow_backend import shape
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Input
from keras.initializations import glorot_uniform as xavier_init
from keras.optimizers import SGD

from deepsign.rp.ri import Generator as RIGen
import numpy as np

import keras.backend as K
import tensorflow as tf

def to_tensor(x):
    x = np.asmatrix(x)
    x = tf.convert_to_tensor(x,dtype=tf.float32)
    return x





# random index dimension
k = 10
s = 2
h_dim = 4
ri_gen = RIGen(active=s,dim=k)

r1 = ri_gen.generate()
print(str(r1))


input = Input(shape=(k,), name="input")
w = Dense(h_dim, activation="linear", init=xavier_init, name="F")(input)
out = Dense(k,activation="sigmoid",init=xavier_init, name="Y")(w)
model = Model(input=input,output=out)

model.summary()

# compile model
sgd = SGD(lr=0.01, decay=1e-5)
model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['binary_accuracy'])

# run one example


x = to_tensor(r1.to_vector())
y = to_tensor(r1.get_positive_vector())

model.fit(x=x,y=y,nb_epoch=1,verbose=1,batch_size=1)


#y = model(x)
#print(K.eval(y))








#features = Dense(h_dim, init=xavier_init, name="features")(input)
#h = Activation("linear", name="h")(features)

# splits into two outputs
#pos_out = Dense(k, activation="sigmoid", init=xavier_init, name="pos_out")(h)
#neg_out = Dense(k, activation="sigmoid", init=xavier_init, name="neg_out")(h)

#model = Model(input=input, output=[pos_out,neg_out])




# we need to determine the appropriate lr
#sgd = SGD(lr=0.01, decay=1e-5)

# the loss is the sum of the two binary cross entropy
#model.compile(optimizer=sgd,
#              loss='binary_crossentropy',
#              metrics=['binary_accuracy'])





#test
#input
#x = ri_0.to_vector()
#x = np.asmatrix(x)

#output
#pos = np.zeros(k)
#neg = np.zeros(k)
#pos[ri_0.positive] = 1
#neg[ri_0.negative] = -1

#pos = np.asmatrix(pos)
#neg = np.asmatrix(neg)



#model.fit(np.transpose(np.asmatrix(x)), [pos, neg], verbose=1, nb_epoch=1)



