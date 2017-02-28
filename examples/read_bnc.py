import os
from deepsign.io.corpora.bnc import BNCReader
from deepsign.io.corpora.bnc import file_walker as luke

home = os.getenv("HOME")
bnc_dir = home+"/data/datasets/bnc"

# e.g. bnc download and unpacked directory
bnc_source_dir = bnc_dir + "/Texts"


file_count = 0
files = luke(bnc_source_dir)

files = iter(sorted(files))

file1 = next(files)
print(file1)

reader = BNCReader(file1)

for i in range(20):
    sentence = next(reader)
    print(sentence)


