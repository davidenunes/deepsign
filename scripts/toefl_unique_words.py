#!/usr/bin/python3

import sys
import os.path

if len(sys.argv) != 2:
    print("argument missing: filename?")
else:
    fname = sys.argv[1]
    if not os.path.isfile(fname):
        print("file not found:", fname)
    else:
        f = open(fname, 'r')
        words = set()
        for line in f:
            tokens = line.split()
            if len(tokens) == 2:
                words.add(tokens[1])
        f.close()

        print("words:")
        fname_out = "TOEFL_words.csv"
        f = open(fname_out, 'w')
        for w in words:
            f.write(w)
            f.write('\n')

        f.close()
