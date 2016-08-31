from deepsign.rp import ri
import numpy as np
import os.path
import zarr


home = os.getenv("HOME")
result_path = home+"/data/results/"


ri_dim = 500
ri_active = 5
num_samples = 10


gen = ri.RandomIndexGenerator(dim=ri_dim, active=ri_active)
c_matrix = np.matrix([gen.generate().to_vector() for i in range(num_samples)])

# write the matrix to hdf5 dataset
sample_word = "test"
vocab = [sample_word] * num_samples
vocab = np.array(vocab)

counts = [1] * num_samples
counts = np.array(counts)


vocab_zarr = zarr.open_like(vocab, path=result_path+'example.zarr', mode='w', chunks=(1000,),
                            compression='blosc',
                            compression_opts=dict(cname='zlib', clevel=3, shuffle=0))
vocab_zarr[:] = vocab

print(vocab_zarr)

for s in vocab_zarr[0:5]:
    print(s)

