import os
import numpy as np
from deepsign.rp.index import TrieSignIndex as Index
from deepsign.io.gold_standards.toefl import TOEFLReader
import seaborn as sns
import matplotlib.pyplot as plt

# model dir
home = os.getenv("HOME")
data_dir = home + "/data/gold_standards/"
result_dir = home + "/data/results/"
model_dir = result_dir + "nrp/300d_reg_embeddings/"
model_file = model_dir + "model_bnc"
embeddings_file = model_dir + "embeddings.npy"
index_file = model_dir + "index.hdf5"

# load index
print("loading word index")
index = Index.load(index_file)
print(len(index))

# load toefl
questions_file = data_dir + "toefl/questions.csv"
answers_file = data_dir + "toefl/answers.csv"

toefl = TOEFLReader(questions_file=questions_file, answers_file=answers_file)

# words in toelf and not in index
toefl_remove = set(w for w in toefl.words if not index.contains(w))
for (i, question) in enumerate(toefl.questions):
    qw = question[0]
    aw = question[1]
    # print(question)
    answer = toefl.answer(i)
    # print(aw[answer])

    words = set([qw] + aw)
    # remove questions that contain words that are not in index
    if not words.isdisjoint(toefl_remove):
        pass

# load embeddings
embeddings = np.load(embeddings_file)
print(embeddings.shape)

riv = index.get_ri("lisbon").to_vector()
print(riv.shape)

lx_vector = np.matmul(riv, embeddings)
print(lx_vector.shape)

sns.set(style="white", context="talk")
sns.distplot(lx_vector);
