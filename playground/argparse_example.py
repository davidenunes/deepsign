import argparse


parser = argparse.ArgumentParser(description="Neural Random Projections arguments")
parser.add_argument('-ri_k',dest="ri_k",type=int,default=1000)
parser.add_argument('-ri_s',dest="ri_s",type=int,default=10)
parser.add_argument('-h_dim',dest="h_dim",type=int,default=300)
parser.add_argument('-window_size',dest="window_size",type=int,default=2)
parser.add_argument('-subsampling',dest="subsampling",type=bool,default=True)
parser.add_argument('-freq_cut',dest="freq_cut",type=float,default=pow(10, -4))
parser.add_argument('-batch_size',dest="batch_size",type=int,default=50)
parser.add_argument('-out_dir',dest="out_dir",type=str,default="/data/results/")


args = parser.parse_args()
print(args)