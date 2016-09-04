import h5py
from deepsign.utils.views import divide_slice
import numpy as np
from tqdm import tqdm


def expand(dataset, n_rows):
    """Expands (resizes) a given dataset by n_rows
    :param dataset a h5py dataset to be resized
    :param n_rows number of rows to be added to the dataset
    """
    dataset.resize(dataset.shape[0] + n_rows, 0)


def batch_write(dataset, data_gen, n_rows, batch_size, progress=False):
    n_batches = 1
    if batch_size < n_rows:
        n_batches = n_rows // batch_size

    # list of ranges that divide the dataset
    ss = divide_slice(n_rows, n_batches)

    batch_progress = range(n_batches)
    if progress:
        batch_progress = tqdm(batch_progress,desc="batch writing")

    for bi in batch_progress:
        current_range = ss[bi]
        current_batch = np.array([next(data_gen) for i in current_range])
        current_slice = slice(current_range.start,current_range.stop,1)
        dataset[current_slice] = current_batch


def str_dataset(hdf5_file, name, shape=(1,), maxshape=(None,)):
    str_type = h5py.special_dtype(vlen=str)
    dataset = hdf5_file.create_dataset(name, shape=shape, maxshape=maxshape, dtype=str_type, compression='gzip')

    return dataset






