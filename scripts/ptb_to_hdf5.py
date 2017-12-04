import sys
import os
from tqdm import tqdm
import numpy as np
import h5py
import argparse
import marisa_trie
from collections import Counter

from deepsign.io.corpora.ptb import PTBReader

parser = argparse.ArgumentParser(description="PTB n-grams to hdf5")
parser.add_argument('-n', dest="n", type=int, default=100)
parser.add_argument('-data_dir', dest="data_dir", type=str, default=os.getenv("HOME")+"/data/datasets/ptb")
parser.add_argument('-out_dir', dest="out_path", type=str, default=os.getenv("HOME")+"/data/datasets")
parser.add_argument('-out_filename', dest="out_filename", type=str, default="ptb_hdf5")
args = parser.parse_args()


hdf5_file = os.path.join(args.data_dir, args.out_filename)

# ======================================================================================
# Build Vocabulary
# the dataset is quite small so we read everything first to build the vocab
# we then use that vocabulary to encode each n-gram
# ======================================================================================
ptb_reader = PTBReader(args.data_dir)

# load vocabulary
word_counter = Counter()
for words in ptb_reader.full():
    word_counter.update(words)

vocab = marisa_trie.Trie(word_counter.keys())


sorted_counts = word_counter.most_common()
word_list, word_freq= zip(*sorted_counts)

#vocabulary = np.array([freq[i][0].encode("utf8") for word in word_list])
vocabulary = np.array(word_list)
frequencies = np.array(word_freq)


dt = h5py.special_dtype(vlen=str)
output_hdf5.create_dataset("vocabulary", data=vocabulary, dtype=dt, compression="gzip")


"""
# create vocab in hdf5


        dt = h5py.special_dtype(vlen=str)
        output_hdf5.create_dataset("vocabulary", data=vocabulary, dtype=dt, compression="gzip")
        print("vocabulary written")

        freq = np.array([freq[i][1] for i in range(len(freq))])
        output_hdf5.create_dataset("frequencies", data=freq, compression="gzip")
        print("frequencies written")

        output_hdf5.close()
        print("done")

"""

"""
if not os.path.isdir(bnc_dir):
    sys.exit("No such directory: {}".format(bnc_dir))


def hdf5_append(sentence):
    #Appends to an hdf5 dataset, duplicates size if full
    global num_rows
    dataset[num_rows] = sentence
    num_rows += 1

    if num_rows == len(dataset):
        dataset.resize(len(dataset) + EXPAND_HDF5_BY, 0)


def hdf5_clean():
    dataset.resize(num_rows, 0)
    h5f.close()


def convert_file(filename):
    reader = BNCReader(filename)

    global max_sentences
    global num_sentences


    for sentence in reader:
        if max_sentences is not None and num_sentences >= max_sentences:
                break
        if len(sentence) > 1:
            s = " ".join(sentence)
            hdf5_append(s)
            num_sentences +=1

    reader.source.close()



print("Processing BNC corpus files in ",bnc_dir)
files = file_walker(bnc_dir)

for file in tqdm(sorted(files)):
        convert_file(file)
        if max_sentences is not None and num_sentences >= max_sentences:
            break
hdf5_clean()

"""



