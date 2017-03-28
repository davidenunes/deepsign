"""
Using mpi4py to train tensorflow models in parallel Asynchronously
"""
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


if rank == 0:
    print("master")
else:
    print("slave")