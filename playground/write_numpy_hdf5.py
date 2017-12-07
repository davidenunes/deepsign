import os.path

import h5py
import numpy as np

from deepsign.rp import ri
from deepsign.utils import h5utils

home = os.getenv("HOME")
result_path = home+"/data/results/"


ri_dim = 500
ri_active = 5
num_samples = 100


gen = ri.Generator(dim=ri_dim, active=ri_active)
c_matrix = np.matrix([gen.generate().to_vector() for i in range(num_samples)])

# write the matrix to hdf5 dataset
sample_word = "test".encode("utf8")
vocab = [sample_word] * num_samples
vocab = np.array(vocab)

counts = [1] * num_samples
counts = np.array(counts)


filename = "random_indexing_test.hdf5"
dataset_path = result_path+filename
print("writing to ",dataset_path)

dataset_name = "ri_d{0}_a{1}".format(ri_dim,ri_active)
print("dataset: "+dataset_name)

h5f = h5py.File(dataset_path,'w')
dt = h5py.special_dtype(vlen=str)

vocabulary_data = h5f.create_dataset("vocabulary", data= vocab, dtype=dt, compression="gzip")
print("vocabulary data written")

count_data = h5f.create_dataset("frequencies", data=counts, compression="gzip")
print("count data written")

ri_data = h5f.create_dataset(dataset_name, data=c_matrix, compression="gzip")
print("random index vectors written")

sum_vectors = h5f.create_dataset(dataset_name+"_sum", data=c_matrix, compression="gzip")
print("random index sum vectors written")

dset0 = h5f.create_dataset("test init",shape=(0,gen.dim),maxshape=(None,gen.dim))
print("0 shape dataset length:", len(dset0))

h5utils.expand(dset0, 1)
print("0 shape dataset length:", len(dset0))
print("len after expand:", len(dset0))
#print("contains",dset0[0])
dset0[0] += gen.generate().to_vector()
print(dset0[0])


h5f.close()