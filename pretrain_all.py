import argparse
import numpy as np
import random
import torch
import mlflow
import time
from pretrainData import PretrainTableDataset
from learning.pretrain import train
from Encodings import table_features
from Utils import subjectColDetection

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="starmie")  # Valerie starmie
    parser.add_argument("--dataset", type=str, default="WDC")  # TabFact open_data WDC
    parser.add_argument("--logdir", type=str, default="model/")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--size", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--lm", type=str, default='sbert')
    parser.add_argument("--pretrain", dest="pretrain", action="store_true")
    parser.add_argument("--NoContext", dest="NoContext", action="store_true")
    parser.add_argument("--projector", type=int, default=768)
    parser.add_argument("--augment_op", type=str, default='sample_cells_TFIDF,sample_cells_TFIDF,sample_cells_TFIDF,sample_cells_TFIDF,sample_cells_TFIDF,sample_cells_TFIDF')  #
    parser.add_argument("--save_model", dest="save_model", action="store_true", default=True)
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    # column header-only mode without table context
    parser.add_argument("--header", dest="header", action="store_true")
    parser.add_argument("--pos_pair", type=int, default=0)
    # single-column mode without table context
    parser.add_argument("--single_column", dest="single_column", action="store_true")
    parser.add_argument("--column", dest="column", action="store_true")
    # subject-column mode without table context
    parser.add_argument("--subject_column", dest="subject_column", action="store_true")
    parser.add_argument("--check_subject_Column", type=str, default='none')  # subjectheader
    # row / column-ordered for preprocessing
    parser.add_argument("--table_order", type=str, default='column')  # column pure_row
    # for sampling
    parser.add_argument("--sample_meth", type=str, default='head')  # head tfidfrow tfidf_cell
    # mlflow tag
    parser.add_argument("--mlflow_tag", type=str, default=None)
    parser.add_argument("--datasetSize", type=int, default=-1)
    parser.add_argument("--deepjoin", dest="deepjoin", action="store_true")
    parser.add_argument("--DPpath", type=str, default="model/Deepjoin/WebData/1600/fineTuneSBERT")


    hp = parser.parse_args()

    # mlflow logging
    for variable in ["method", "batch_size", "lr", "n_epochs", "augment_op", "sample_meth", "table_order"]:
        mlflow.log_param(variable, getattr(hp, variable))

    if hp.mlflow_tag:
        mlflow.set_tag("tag", hp.mlflow_tag)

    # set seed
    seed = hp.run_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Change the data paths to where the benchmarks are stored
    subjectColDetection('datasets/%s/' % hp.dataset)
    path = 'datasets/%s/Test' % hp.dataset
    start_time_preprocess = time.time()

    trainset = PretrainTableDataset.from_hp(path, hp)
    #print(len(trainset),trainset[0])
    end_time_preprocess = time.time()
    time_difference_pre = end_time_preprocess - start_time_preprocess
    print(f"Preprocess time: {time_difference_pre}\n")

    output_path = 'result/embedding/%s' % hp.dataset
    if hp.pretrain:
        start_time_pretrain = time.time()
        trainset.encodings(output_path, setting=hp.NoContext)
        end_time_pretrain = time.time()
        time_difference_pretrain = end_time_pretrain - start_time_pretrain
        print(f"Encode time: {time_difference_pretrain}\n ")
    elif hp.deepjoin:
        start_time_pretrain = time.time()
        trainset.encodings(output_path, setting=hp.NoContext)
        end_time_pretrain = time.time()
        time_difference_pretrain = end_time_pretrain - start_time_pretrain
        print(f"Encode time: {time_difference_pretrain}\n ")
    else:

        start_time_train = time.time()
        train(trainset, hp)
        end_time_train = time.time()
        time_difference_train = end_time_train - start_time_train
        print(f"Train time: {time_difference_train}\n")

        start_time_encode = time.time()
        table_features(hp)
        end_time_encode = time.time()
        time_difference_encode = end_time_encode - start_time_encode
        print(f"Encode time: {time_difference_encode} \n ")
