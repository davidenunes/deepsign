#!/usr/bin/env python

""" Run with
    mpirun -ngram_size 10 python -embed_size scripts.parallel.word_frequency
"""
import os
import time

import h5py
import numpy as np
from mpi4py import MPI
from tqdm import tqdm

from deepsign.data.corpora.pipe import BNCPipe
from deepsign.utils.views import divide_slice, subset_chunk_it

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

home = os.getenv("HOME")
data_dir = home + "/data/gold_standards/"
corpus_file = data_dir + "bnc.hdf5"


class Tags():
    FINISHED = 0
    RESULT = 1



size = comm.size
num_slaves = size-1
num_rows = 1000

# ======================================================================================
# Master Node
# ======================================================================================
if comm.rank == 0:

    # open hdf5 file and get the dataset
    corpus_hdf5 = h5py.File(corpus_file, 'r')
    corpus_dataset = corpus_hdf5["sentences"]

    if num_rows == -1:
        num_rows = len(corpus_dataset)


    print("Master Node: preparing data, processing [ %d of %d ]"%(num_rows,len(corpus_dataset)))

    subset_slices = divide_slice(n=num_rows, n_slices=num_slaves)

    print("Sending Tasks to %d nodes"%(num_slaves))
    # send slices
    for node in range(1,size):
        slice_i = node-1
        comm.send(subset_slices[slice_i],dest=node)

    print("Data Delivered")
    num_finished = 0

    while num_finished < num_slaves:
        print("Waiting for results...")
        status = MPI.Status()
        data = comm.recv(source=MPI.ANY_SOURCE, status=status)
        tag = status.Get_tag()
        source = status.Get_source()
        if tag == Tags.FINISHED:
            print("Node %d Finished" % (source))
            num_finished += 1
        else:
            pass
            #print("Received from %d: " % (source), data)

    print("All Done in root!")

# ======================================================================================
# Slave Node
# ======================================================================================
else:
    subset_slice = comm.recv(source=0)
    print("Node %d: Processing slice: " % comm.rank, str(subset_slice))

    # open hdf5 file and get the dataset
    corpus_hdf5 = h5py.File(corpus_file, 'r')
    corpus_dataset = corpus_hdf5["sentences"]

    sentences = subset_chunk_it(corpus_dataset, subset_slice, chunk_size=1000)
    pipeline = BNCPipe(datagen=sentences, lemmas=True)

    for sentence in tqdm(pipeline,total=len(subset_slice)):
        #print("Node %d: " % comm.rank, sentence)
        pass

    dummy_results = np.arange(10, dtype='i')
    comm.send(dummy_results, dest=0, tag=Tags.RESULT)
    time.sleep(2)
    comm.send("Done", dest=0, tag=Tags.FINISHED)

    corpus_hdf5.close()
