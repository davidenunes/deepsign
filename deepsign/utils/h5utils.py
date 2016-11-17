import h5py
from deepsign.utils.views import divide_slice
import numpy as np
from tqdm import tqdm


def extend_rows(dataset, n_rows):
    """Expands (resizes) a given dataset by n_rows
    :param dataset a h5py dataset to be resized
    :param n_rows number of rows to be added to the dataset
    """
    dataset.resize(dataset.shape[0] + n_rows, 0)


def update_dataset(dataset, data_dict, update_fn=None):
    """Updates the given dataset for each row i that is a key in the data_dict
    if the update function is None it replaces the respective entries in the dataset with
    the entries in the data_dict

    if the biggest integer key us bigger than the dataset number of rows, the dataset is extended,
    note that the dataset must be extensible to begin with.

    :param dataset: a given hdf5 dataset
    :param data_dict: a dictionary index -> data, the data must be in vector form and with a shape matching the dataset
    rows
    :param update_fn: update function that takes an entry and updates the current dataset row 
    """
    data_i = sorted(data_dict.keys())
    max_i = data_i[-1]

    if max_i >= len(dataset):
        add_nrows = max_i - len(dataset) + 1
        extend_rows(dataset, add_nrows)

    # update dataset entries
    for i in data_i:
        if update_fn is None:
            dataset[i] = data_dict[i]
        else:
            dataset[i] = update_fn(dataset[i], data_dict[i])


def batch_write(dataset, data_gen, n_rows, batch_size, progress=False):
    """
    Writes data to given dataset (hdf5 or equivalent handle) in batches

    :param dataset: a given dataset where the data will be written to
    :param data_gen: the data generator containing the data to be written
    :param n_rows: number of data rows to be written
    :param batch_size: size of the batch to be written which dictates how many items we take from data_gen at a time
    :param progress: True if we want to track progress
    """
    n_batches = 1
    if batch_size < n_rows:
        n_batches = n_rows // batch_size

    # list of ranges that divide the dataset
    ss = divide_slice(n_rows, n_batches)

    batch_progress = range(n_batches)
    if progress:
        batch_progress = tqdm(batch_progress, desc="batch writing")

    for bi in batch_progress:
        current_range = ss[bi]
        current_batch = np.array([next(data_gen) for i in current_range])
        current_slice = slice(current_range.start, current_range.stop, 1)
        dataset[current_slice] = current_batch


def str_dataset(hdf5_file, name, shape=(1,), maxshape=(None,)):
    str_type = h5py.special_dtype(vlen=str)
    dataset = hdf5_file.create_dataset(name, shape=shape, maxshape=maxshape, dtype=str_type, compression='gzip')

    return dataset





