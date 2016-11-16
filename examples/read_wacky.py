import sys
import os
import fnmatch
import re
from deepsign.io.corpora.wacky import WaCKyCorpus



dataset_name = "sentences"
lemmatize = False
max_sentences = 100
dir="/home/davex32/Dropbox/research/Data/WaCKy"




file_number_re = re.compile('(\d{1,2})')

if not os.path.isdir(dir):
    sys.exit("No such directory: {}".format(dir))


def read_file(file_name, max_sentences):
    file_name = os.path.join(dir, file_name)
    print("Reading %s into" % (file_name))
    reader = WaCKyCorpus(file_name, lemmatize)

    for i in range(max_sentences):
        sentence = next(reader)
        if len(sentence) > 1:
            s = " ".join(sentence)
            print(s)

    reader.source.close()
    print("Finished with: ", file_name)


print("Processing WaCKy corpus files in ",dir)
files = [f for f in os.listdir(dir) if fnmatch.fnmatch(f,"*.xml.gz")]

for file in files:
    read_file(file,max_sentences)


