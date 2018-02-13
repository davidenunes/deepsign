import importlib
from exp.params import ParamSpace
from exp.grid import GridRunner, GridConf
import argparse
import os

# directly importing this trows an error
# because python doesn't know that the current folder
grid_config = importlib.import_module("grid_config")

parser = argparse.ArgumentParser(description="NNLM Baseline Parameters")
parser.add_argument('-grid', choices=['mas', 'ingrid'], default="mas")
parser.add_argument('-run', type=str, default="run/nnlm.py")
args = parser.parse_args()

# load grid config
grid_cfg = getattr(grid_config, args.grid)
print(grid_cfg)

# check if script exists
if not os.path.exists(args.run):
    raise ValueError("\"run\" script not found")

# create ParamSpace
ps = ParamSpace("baseline.params")
ps.add_list("num_h", [1, 2, 3])


runner = GridRunner(grid_cfg)
runner.submit_all(args.run, ps, "baseline", None, call_qsub=False)
