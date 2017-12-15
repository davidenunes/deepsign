import argparse


parser = argparse.ArgumentParser(description="testing argparse")
parser.add_argument('-ri_k', type=int, default=1000)
args = parser.parse_args()
print(args)
