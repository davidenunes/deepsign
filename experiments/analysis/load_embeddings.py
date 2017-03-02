import os
import tensorflow as tf

# model dir
home = os.getenv("HOME")
result_dir = home + "/data/results/"
model_dir = result_dir + "nrp_300d_reg_embeddings/"
model_file = model_dir + "model_bnc"

