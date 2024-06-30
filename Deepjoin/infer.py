import argparse

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="dataset")
parser.add_argument("--datafile", help="datafile")
parser.add_argument("--storepath", help="storepath")

args = parser.parse_args()

dataset  = args.dataset_para
datafile = args.datafile
storepath= args.storepath
if __name__ == '__main__':
 print("TBC")

