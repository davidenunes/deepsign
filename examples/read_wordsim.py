#!/usr/bin/python3

import os.path
from deepsign.io.datasets.wordsim import WordSim353Reader as WS


home = os.getenv("HOME")
wordsim_dir = home+"/data/datasets/wordsim/"
sim = open(wordsim_dir+"sim.csv", 'r')
rel = open(wordsim_dir+"rel.csv", 'r')


reader = WS(sim,rel)
print(reader.rel)

sim.close()
rel.close()