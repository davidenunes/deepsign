# This script iterates over the wacky corpus using the wacky corpus reader
# and returns a new version of the corpus where each file has one sentence per line
# and is encoded as utf-8. The idea is to make it easier for parallel processing of this corpus
#
# Copyright 2016 Davide Nunes

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ======================================================================================================================

import sys
import os
import fnmatch
import re
from tqdm import tqdm
import h5py
from deepsign.io.corpora.wacky import WaCKyCorpus


dir="/home/davex32/Dropbox/research/Data/WaCKy"
output_fname = "wacky_10M.hdf5"
lemmatize = False
# global sentence count

max_sentences = 10000000
num_sentences = 0
pbar = tqdm(total=max_sentences)


h5f_name = os.path.join(dir,output_fname)
dataset_name = "sentences"

# open hdf5 file and create dataset
h5f = h5py.File(h5f_name, "a")
# store strings as variable-length UTF-8
dt = h5py.special_dtype(vlen=str)
dataset = h5f.create_dataset(dataset_name, (1,), maxshape=(None,), dtype=dt, compression="gzip")
num_rows = 0
EXPAND_HDF5_BY = 1000



base_filename = ""
file_number_re = re.compile('(\d{1,2})')


if not os.path.isdir(dir):
    sys.exit("No such directory: {}".format(dir))


def hdf5_append(sentence):
    """Appends to an hdf5 dataset, duplicates size if full"""
    global num_rows
    dataset[num_rows] = sentence
    num_rows += 1

    if num_rows == len(dataset):
        dataset.resize(len(dataset) + EXPAND_HDF5_BY, 0)


def hdf5_clean():
    dataset.resize(num_rows, 0)
    h5f.close()


def convert_file(file_name):
    file_name = os.path.join(dir, file_name)
    print("Converting %s into %s" % (file_name, h5f_name))
    reader = WaCKyCorpus(file_name, lemmatize)

    global max_sentences
    global num_sentences
    global pbar

    for sentence in reader:
        if max_sentences is not None and num_sentences >= max_sentences:
                break
        if len(sentence) > 1:
            s = " ".join(sentence)
            hdf5_append(s)
            num_sentences +=1
            pbar.update(1)

    reader.source.close()
    print("Finished with: ", file_name)

print("Processing WaCKy corpus files in ",dir)
files = [f for f in os.listdir(dir) if fnmatch.fnmatch(f,"*.xml.gz")]




# convert each file
for file in files:
        convert_file(file)
        if max_sentences is not None and num_sentences >= max_sentences:
            break
hdf5_clean()





