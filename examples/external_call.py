import sys
import os
from subprocess import run

home = os.getenv("HOME")
sys.path.extend([home + '/dev/deepsign', home + '/dev/tensorx', home + '/dev/params'])

#print(sys.path)
venv = home + "/dev/.envs/deepsign/bin/activate"
venv_python = home + "/dev/.envs/deepsign/bin/python"

#print("venv: ", venv)


run([venv_python, home+"/dev/deepsign/examples/test_tokenizer.py"])
