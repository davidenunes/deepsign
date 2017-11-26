import os


class PTBIterator:
    """
    Simple iterator, the file since the file has one sentence per line
    """
    def __init__(self, path):
        assert os.path.exists(path)
        self.path = path
        self.current_sentence = None
        self.source = open(path,'r')

    def close(self):
        self.file.close()

    def __iter__(self):
        return self

    def __next__(self):
        self.current_sentence = self.source.readline()
        if not self.current_sentence:
            self.source.close()
            raise StopIteration

        return self.current_sentence


class PTBReader:
    UNKNOWN_TOKEN = "<unk>"

    """ PTB Corpus Reader

        Implements a sentence iterator over PTB WSJ corpus files. Provided by Mikolov
        with the same pre-processing as in the paper:
         "Empirical Evaluation and Combination of Advanced Language Modeling Techniques"

        This allows for:
            Iterate over the corpus returning sentences in the form of lists of strings
            
        Provides access to iterators for each section of the corpus: full, train, valid, test.
        the train, valid, and test sets are the same as in the paper. 
        
        Splits:
            Sections 0-20 were used as training data (930k tokens), sections 21-22 as validation 
            data (74k tokens) and 23-24 as test data (82k tokens).
            
        Vocab:
            Vocabulary is fixed to 10k unique tokens, words outside this vocabulary are set to
            PTBReader.UNKNOWN_TOKEN
            
        Args:
            path: path to the directory containing the dataset files
    """

    def __init__(self, path):
        self.train_fn = os.path.join(path, 'train.txt')
        self.valid_fn = os.path.join(path, 'valid.txt')
        self.test_fn = os.path.join(path, 'test.txt')

        assert os.path.exists(self.train_fn)
        assert os.path.exists(self.valid_fn)
        assert os.path.exists(self.test_fn)

    def training_set(self):
        return PTBIterator(path=self.train_fn)

    def validation_set(self):
        return PTBIterator(path=self.valid_fn)

    def test_set(self):
        return PTBIterator(path=self.test_fn)

