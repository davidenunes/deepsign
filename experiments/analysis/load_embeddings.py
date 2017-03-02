import os
import tensorflow as tf
from tensorx.models.nrp import NRP
import numpy as np
from deepsign.rp.index import TrieSignIndex as Index
from deepsign.io.datasets.toefl import TOEFLReader


# model dir
home = os.getenv("HOME")
data_dir = home + "/data/datasets/"
result_dir = home + "/data/results/"
model_dir = result_dir + "nrp/nrp_300d_reg_embeddings/"
model_file = model_dir + "model_bnc"
index_file = model_dir + "index.hdf5"




# load index
print("loading word index")
index = Index.load(index_file)
print(len(index))



# load toefl
questions_file = data_dir + "toefl/questions.csv"
answers_file = data_dir + "toefl/answers.csv"

toefl = TOEFLReader(questions_file=questions_file,answers_file=answers_file)

# words in toelf and not in index
toefl_remove = set(w for w in toefl.words if not index.contains(w))
for (i,question) in enumerate(toefl.questions):
    qw = question[0]
    aw = question[1]
    #print(question)
    answer = toefl.answer(i)
    #print(aw[answer])

    words = set([qw] + aw)
    # remove questions that contain words that are not in index
    if not words.isdisjoint(toefl_remove):
        pass




# load model
print("loading model")
k = 1000
h_dim = 300
model = NRP(k_dim=k, h_dim=h_dim)

tf_session = tf.Session()

model.load(tf_session,model_file)

w = tf_session.run(model.h.weights)
print(np.max(w))
print(np.min(w))


tf_session.close()
