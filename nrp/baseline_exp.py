import sys

import grid_confs
import baseline_params
from exp.grid import GridRunner
import argparse

# TODO make a runner as a cli app for the exp library

parser = argparse.ArgumentParser(description="NNLM Baseline Parameters")
parser.add_argument('-grid', choices=['mas', 'ingrid'], default="mas")
parser.add_argument('-run_script', dest="run_script", type=str, default="run/nnlm.py")
parser.add_argument('-params', dest="params", type=str, default="baseline")
args = parser.parse_args()

grid_cfg = getattr(grid_confs, args.grid)
print(grid_cfg)
param_space = getattr(baseline_params, args.params)
print(param_space)

runner = GridRunner(grid_cfg)
runner.submit_all(args.run_script, param_space, "baseline", None, call_qsub=False)
