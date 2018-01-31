import argparse
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

parser = argparse.ArgumentParser(description="testing argparse")
parser.add_argument('-ri_k', type=int, default=1000)
parser.add_argument('-ri_s', type=float, default=0.1)
parser.add_argument('-f', type=str)
args = parser.parse_args()

print(args)


arg_dict = vars(args)

print(arg_dict)
