#!/usr/bin/env python

# Run with
# mpirun -n 10 python -m scripts.parallel.word_frequency

from mpi4py import MPI

import time
from deepsign.utils.views import divide_slice
import numpy
import h5py
import sys

#TODO not finished
hdf5_file_path = sys.argv[1]

num_records = -1
if len(sys.argv) >2:
    num_records = int(sys.argv[2])

dataset_name = "ukwac_sentences"

class Tags():
    FINISHED = 0
    RESULT = 1


comm = MPI.COMM_WORLD


size = comm.size
num_slaves = size-1

if comm.rank == 0:
    # master node
    # open hdf5 file and get the dataset
    f = h5py.File(hdf5_file_path, 'a')
    dataset = f[dataset_name]

    if num_records == -1:
        num_records = len(dataset)


    print("Master Node: preparing data, processing [ %d of %d ]"%(num_records,len(dataset)))

    slices = divide_slice(num_elems=num_records, num_slices=num_slaves)

    print("Sending Tasks to %d nodes"%(num_slaves))
    # send slices
    for node in range(1,size):
        slice_i = node-1
        comm.send(slices[slice_i],dest=node)

else:
    # slave nodes
    slice = comm.recv(source=0)
    print("Node %d has slice: " % comm.rank, slice)

# wait for everyone to receive data (not needed)
comm.Barrier()

if comm.rank == 0:
    print("Data Delivered")
    num_finished = 0

    while num_finished < num_slaves:
        print("Waiting for results...")
        status = MPI.Status()
        data= comm.recv(source=MPI.ANY_SOURCE,status=status)
        tag = status.Get_tag()
        source = status.Get_source()
        if tag == Tags.FINISHED:
            print("Node %d Finished"%(source))
            num_finished += 1
        else:
            print("Received from %d: "%(source),data)

    print("All Done")

else:
    print("Node %d: Processing slice: " % comm.rank, slice)

    # open hdf5 file and get the dataset
    f = h5py.File(hdf5_file_path,'r')
    dataset = f[dataset_name]
    dataset = dataset[slice]

    for s in range(len(dataset)):
        print("Node %d: " % comm.rank, dataset[s][0].split(" ")[0:3])

    dummy_results = numpy.arange(10,dtype='i')
    comm.send(dummy_results,dest=0, tag=Tags.RESULT)
    time.sleep(2)
    comm.send("Done",dest=0, tag=Tags.FINISHED)

    f.close()
