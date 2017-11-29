#!/usr/bin/env python

""" Run with
    mpirun -ngram_size 10 python -embed_size scripts.parallel.word_frequency

    Using mpi4py to train tensorflow models in parallel Asynchronously
"""
from mpi4py import MPI
from deepsign.utils.views import divide_slice, subset_chunk_it


class Tags:
    # task has finished
    FINISHED = 0
    # report task progress (IT,MAX_IT,ETA)
    PROGRESS = 1
    # pull var from master (var,ids)
    # push var to server (var, ids, values)

    # for synchronous sync we just need all the variables in all the nodes
    # and calling sync prompts the master to call reduce all in a specific order
    # same order as the task nodes
    # so we have to supply the sync routine where we have a list of vars and
    # update them all
    """
    An idea could be to reduce sum a sparse vector with all the ids for the rows to be 
    updated, get the non-zero entries to get the ids and update them in the given order
    """


class MPITrainer:
    """The idea is for this class to be initialised on each node and it is used to organise
    the code for each rank.
    """

    MASTER_RANK = 0

    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.data_it = None

    def is_master(self):
        return self.comm.rank == MPITrainer.MASTER_RANK

    def split_data(self, dataset, chunk_size=100):
        if not self.is_master():
            current_rank = self.rank
            num_rows = len(dataset)
            # the master node manages the parallel process and parameter sync
            num_tasks = self.size - 1
            subset_slices = divide_slice(n=num_rows, n_slices=num_tasks)
            current_slice = subset_slices[current_rank - 1]
            # this iterator loads chunk_size at a time, but we still iterate over each example
            self.data_it = subset_chunk_it(dataset, current_slice, chunk_size)



