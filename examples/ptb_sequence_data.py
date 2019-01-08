import collections
import os
import sys
from deepsign.data import iterators as it
from deepsign.data.corpora.ptb import PTBReader
import numpy as np
import h5py
import marisa_trie

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf


def get_batch(generator, batch_size, num_steps, max_word_length, pad=False):
    """Read batches of input."""
    cur_stream = [None] * batch_size

    inputs = np.zeros([batch_size, num_steps], np.int32)
    char_inputs = np.zeros([batch_size, num_steps, max_word_length], np.int32)
    global_word_ids = np.zeros([batch_size, num_steps], np.int32)
    targets = np.zeros([batch_size, num_steps], np.int32)
    weights = np.ones([batch_size, num_steps], np.float32)

    no_more_data = False
    while True:
        inputs[:] = 0
    char_inputs[:] = 0
    global_word_ids[:] = 0
    targets[:] = 0
    weights[:] = 0.0

    for i in range(batch_size):
        cur_pos = 0

        while cur_pos < num_steps:
            if cur_stream[i] is None or len(cur_stream[i][0]) <= 1:
                try:
                    cur_stream[i] = list(generator.next())
                except StopIteration:
                    # No more data, exhaust current streams and quit
                    no_more_data = True
                    break

            how_many = min(len(cur_stream[i][0]) - 1, num_steps - cur_pos)
            next_pos = cur_pos + how_many

            inputs[i, cur_pos:next_pos] = cur_stream[i][0][:how_many]
            char_inputs[i, cur_pos:next_pos] = cur_stream[i][1][:how_many]
            global_word_ids[i, cur_pos:next_pos] = cur_stream[i][2][:how_many]
            targets[i, cur_pos:next_pos] = cur_stream[i][0][1:how_many + 1]
            weights[i, cur_pos:next_pos] = 1.0

            cur_pos = next_pos
            cur_stream[i][0] = cur_stream[i][0][how_many:]
            cur_stream[i][1] = cur_stream[i][1][how_many:]
            cur_stream[i][2] = cur_stream[i][2][how_many:]

            if pad:
                break

        if no_more_data and np.sum(weights) == 0:
            # There is no more data and this is an empty batch. Done!
            break
        yield inputs, char_inputs, global_word_ids, targets, weights


if __name__ == '__main__':
    data_path = os.path.join(os.getenv("HOME"), "data/datasets/ptb/")

    corpus = PTBReader(path=data_path, mark_eos=True)
    corpus_stats = h5py.File(os.path.join(data_path, "ptb_stats.hdf5"), mode='r')
    vocab = marisa_trie.Trie(corpus_stats["vocabulary"])

    batch_size = 4
    num_steps = 3

    # data = [vocab[word] for word in it.take_it(1000, it.flatten_it(corpus.training_set(1000)))]
    data = [word for word in it.take_it(it.flatten_it(corpus.training_set(1000)), 1000)]
    data = iter((c for c in it.flatten_it(data)))
    print(next(data))
    # data = np.array(data)

    # Starting from sequential data, batchify arranges the dataset into columns.
    # For instance, with the alphabet as the sequence and batch size 4, we'd get
    # ┌ a g m s ┐
    # │ b h n t │
    # │ c i o u │
    # │ d j p v │
    # │ e k q w │
    # └ f l r x ┘.
    # These columns are treated as independent by the model, which means that the
    # dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
    # batch processing.

    # num_batches = len(data) // batch_size

    # print(list(it.take_it(2*batch_size*num_steps,data)))
    # data = it.batch_it(data,num_steps)
    # data = it.batch_it(data,batch_size)
    # print(np.array(next(data)))
    # print(np.array(next(data)))

    # get_batch subdivides the source data into chunks of length args.bptt.
    # If source is equal to the example output of the batchify function, with
    # a bptt-limit of 2, we'd get the following two Variables for i = 0:
    # ┌ a g m s ┐ ┌ b h n t ┐
    # └ b h n t ┘ └ c i o u ┘
    # Note that despite the name of the function, the subdivison of data is not
    # done along the batch dimension (i.e. dimension 1), since that was handled
    # by the batchify function. The chunks are along dimension 0, corresponding
    # to the seq_len dimension in the LSTM.

    # print(list(it.batch_it(it.consume_it(6,train),4)))[0])

    data, *_ = get_batch(data, batch_size, num_steps, max_word_length=4)
    print(np.array(data))
