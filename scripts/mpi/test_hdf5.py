#!/usr/bin/env python

# Run with
# mpiexec -n 2 python -m scripts.parallel.test_hdf5 ~/Dropbox/research/Data/WaCKy/wacky.hdf5

from mpi4py import MPI

import h5py
import sys


dataset_path = sys.argv[1]

dataset_name = "ukwac_sentences"

comm = MPI.COMM_WORLD

if comm.rank == 0:
    # open hdf5 file and get the dataset
    print("reading file: ", dataset_path)

    f = h5py.File(dataset_path,'r')
    dataset = f[dataset_name]

    print(dataset[0])

    f.close()