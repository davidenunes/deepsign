import sys
import os
import fnmatch
import re
from deepsign.data.corpora.wacky import WaCKyCorpus


home = os.getenv("HOME")
dir = home + "/data/gold_standards/wacky"


lemmatize = False
max_sentences = 10


file_number_re = re.compile('(\d{1,2})')

if not os.path.isdir(dir):
    sys.exit("No such directory: {}".format(dir))


def read_file(file_name, max_sentences):
    file_name = os.path.join(dir, file_name)
    #print("Reading %s into" % (file_name))
    reader = WaCKyCorpus(file_name, lemmatize)

    for i in range(max_sentences):
        sentence = next(reader)
        if len(sentence) > 1:
            print(sentence)

    reader.source.close()
    #print("Finished with: ", file_name)


files = [f for f in os.listdir(dir) if fnmatch.fnmatch(f,"*.xml.gz")]

file = files[0]
read_file(file,max_sentences)


