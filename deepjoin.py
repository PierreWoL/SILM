import argparse
import logging
import os
import torch
import pickle
from Deepjoin.train import construct_train_dataset, train_model


parser = argparse.ArgumentParser()

parser.add_argument("--colToText", type=str, default="title-colname-stat-col", help="col to text transformation")
parser.add_argument("--model_name", type=str, default=None, help="Base model name for training")
parser.add_argument("--shuffle_rate", type=float, default=0.2)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--warmup_steps", type=int, default=None)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--num_epochs", type=int, default=3)
parser.add_argument("--local-rank", type=int, default=-1, help="Dummy argument for compatibility with transformers")

logging.basicConfig(level=logging.INFO)
logging.getLogger(__name__).setLevel(logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "all-mpnet-base-v2"  # multi-qa-distilbert-cos-v1
    data_path = "datasets/WDC/Test/"
    naming_file = "datasets/WDC/naming.csv"
    train_dataset = construct_train_dataset(data_path, naming_file, model_name=model_name,
                                            column_to_text_transformation=args.colToText,
                                            shuffle_rate=args.shuffle_rate,  device=device)

    with open(os.path.join( "datasets/WDC/dataset", 'trainDeepJoinDataWDC.pickle'), 'wb') as f:
        pickle.dump(train_dataset, f)
    train_model(model_name=model_name, train_dataset=train_dataset, dev_samples=None, model_save_path="",
                batch_size=args.batch_size,
                learning_rate=args.learning_rate, warmup_steps=args.weight_decay,
                weight_decay=args.weight_decay, num_epochs=args.weight_decay,
                device=device, cpuid=2)