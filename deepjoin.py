import argparse
import logging
import os
import sys

import torch
import pickle
from Deepjoin.train import construct_train_dataset,   train_model
from Utils import mkdir

parser = argparse.ArgumentParser()

parser.add_argument("--colToText", type=str, default="title-colname-stat-col", help="col to text transformation")
parser.add_argument("--datasetSize", type=int, default=800)
parser.add_argument("--model_name", type=str, default="all-mpnet-base-v2", help="Base model name for training")
parser.add_argument("--dataset", type=str, default="WDC", help="used dataset")
parser.add_argument("--shuffle_rate", type=float, default=0.3)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--warmup_steps", type=int, default=None)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--num_epochs", type=int, default=25)
parser.add_argument("--dist_url", type=str, default="tcp://")
parser.add_argument("--world_size", type=int, default=1)
parser.add_argument("--rank", type=int, default=0)
parser.add_argument("--local-rank", type=int, default=-1, help="Dummy argument for compatibility with transformers")

logging.basicConfig(level=logging.INFO)
logging.getLogger(__name__).setLevel(logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = args.dataset
    model_name = "all-mpnet-base-v2"
    data_path = f"datasets/{dataset}/Test/"
    naming_file =""  #f"datasets/{dataset}/naming.csv"
    if os.path.isfile(os.path.join(f"datasets/{dataset}/",f'trainDeepJoinData{dataset}_{args.datasetSize}.pickle')):
        with open(os.path.join(f"datasets/{dataset}/", f'trainDeepJoinData{dataset}_{args.datasetSize}.pickle'), 'rb') as f:
            train_dataset = pickle.load(f)
        print("Read successfully")
    else:
        print("Start constructing training dataset...")
        train_dataset = construct_train_dataset(data_path, naming_file, model_name=model_name,
                                                column_to_text_transformation=args.colToText,
                                                shuffle_rate=args.shuffle_rate, device=device,
                                                select_num=args.datasetSize)
        with open(os.path.join(f"datasets/{dataset}/",
                               f'trainDeepJoinData{dataset}_{args.datasetSize}.pickle'), 'wb') as f:
            pickle.dump(train_dataset, f)
        print("Succeeded in building and saving training dataset...")
    
    path = f"model/Deepjoin/{dataset}/{args.datasetSize}/"
    mkdir(path)
    num_gpus = torch.cuda.device_count()
    cupid = 0
    if num_gpus >= 2:
        cupid = 2
    elif num_gpus ==1:
        cupid = 0
    else:
      print("NO GPU AVAILABLE!")
      sys.exit(1)
      
    train_model(model_name=model_name, train_dataset=train_dataset[:320000], dev_samples=None,
                model_save_path=os.path.join(path, "fineTuneSBERT61.pt"),
                batch_size=args.batch_size,
                warmup_steps=args.warmup_steps, cpuid = cupid,
                weight_decay=args.weight_decay, num_epochs=args.num_epochs,
                device=device)
    