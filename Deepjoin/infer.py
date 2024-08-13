import argparse
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, InputExample

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="GDS")


args = parser.parse_args()


if __name__ == '__main__':
    path = "E:/Project/CurrentDataset/model/Deepjoin/WebData/200/fineTuneSBERT.pt"
    model = SentenceTransformer(path)
    data_name = [i for i in os.listdir(f"E:/Project/CurrentDataset/datasets/{args.dataset}/Test/") if i.endswith(".csv")]
