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
import gzip
from deepsign.io.corpora.wacky import WaCKyCorpus
import re
import multiprocessing


dir = sys.argv[1]
lemma = bool(sys.argv[2])
base_filename = "UKWAC"
file_number_re = re.compile('(\d{1,2})')


if not os.path.isdir(dir):
    sys.exit("No such directory: {}".format(dir))

print("Processing WaCKy corpus files in ",dir)
files = [f for f in os.listdir(dir) if fnmatch.fnmatch(f,"*.xml.gz")]


def convert_file(file):
    file = os.path.join(dir,file)
    number = file_number_re.search(file).group(1)
    new_filename = "{}-{}.txt.gz".format(base_filename, number)
    new_filename = os.path.join(dir,new_filename)
    print("Converting %s into %s"%(file,new_filename))

    reader = WaCKyCorpus(file, lemma)
    writer = gzip.open(new_filename,mode="wt",encoding="utf-8")

    for sentence in reader:
        if len(sentence) > 1:
            s = " ".join(sentence)
            s += "\n"
            writer.write(s)

    reader.source.close()
    writer.close()
    print("File created: ",new_filename)


pool = multiprocessing.Pool(8)

convertfn = convert_file
pool.map(convertfn, files)





