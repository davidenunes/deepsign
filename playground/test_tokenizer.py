import os
import sys

#home = os.getenv("HOME")
#sys.path.extend([home + '/dev/deepsign', home + '/dev/tensorx', home + '/dev/params'])

from deepsign.nlp.tokenization import Tokenizer

sentence = "hello there mr anderson!"

tokenizer = Tokenizer()

tokens = tokenizer.tokenize(sentence)
print(tokens)


