import os
import tensorflow as tf
from tensorx.models.nrp import NRP
import numpy as np


# model dir
home = os.getenv("HOME")
data_dir = home + "/data/datasets/"
result_dir = home + "/data/results/"
model_dir = result_dir + "nrp/300d_reg_all/"
model_file = model_dir + "model_bnc"

weights_file = model_dir + "embeddings.npy"


# load model
print("loading model")
k = 1000
h_dim = 300
model = NRP(k_dim=k, h_dim=h_dim)

tf_session = tf.Session()

model.load(tf_session,model_file)

w = tf_session.run(model.h.weights)

np.save(weights_file,w)



w2 = np.load(weights_file)

np.testing.assert_array_equal(w,w2)



tf_session.close()
