import os
import sys
import h5py
from tqdm import tqdm


from experiments.pipe.bnc_pipe import BNCPipe
from deepsign.utils.views import chunk_it
from deepsign.nlp.tokenization import Tokenizer
from deepsign.rp.index import TrieSignIndex
from deepsign.rp.ri import Generator, RandomIndex


from tensorx.models.nrp import NRP
import tensorflow as tf

#======================================================================================
# Load Corpus
#======================================================================================
home = os.getenv("HOME")
data_dir = home + "/data/datasets/"
corpus_file = data_dir + "bnc_full.hdf5"

corpus_hdf5 = h5py.File(corpus_file, 'r')
corpus_dataset = corpus_hdf5["sentences"]
# iterates over lines but loads them as chunks
sentences = chunk_it(corpus_dataset, chunk_size=40000)

pipeline = BNCPipe(datagen=sentences)
#======================================================================================
# Load Vocabulary
#======================================================================================
vocab_file = data_dir + "bnc_vocab_spacy.hdf5"
vocab_hdf5 = h5py.File(vocab_file, 'r')
k = 1000
s = 10
ri_gen = Generator(dim=k, active=s)
print("Loading Vocabulary...")
sign_index = TrieSignIndex(ri_gen, list(vocabulary[()]))



#======================================================================================
# Process Corpus
#======================================================================================
try:
    for tokens in tqdm(pipeline,total=len(corpus_dataset)):
        pass
        #print(tokens)

    corpus_hdf5.close()
#======================================================================================
# Process Interrupted
#======================================================================================
except (KeyboardInterrupt,SystemExit):
    # TODO store the model current state
    # and the state of the corpus iteration
    print("\nProcess interrupted, closing corpus and saving progress...",file=sys.stderr)
    corpus_hdf5.close()
    vocab_hdf5.close()
else:
    #save the model
    pass

