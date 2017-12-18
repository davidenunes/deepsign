#!/usr/bin/python3

import os.path
from deepsign.data.corpora.toefl import TOEFLReader as TOEFL


home = os.getenv("HOME")
toefl_dir = home+"/data/gold_standards/toefl/"




questions = open(toefl_dir+"questions.csv", 'r')
answers = open(toefl_dir+"answers.csv", 'r')
reader = TOEFL(questions,answers)

questions.close()
answers.close()

