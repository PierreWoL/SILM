import argparse
import numpy as np
import random
import torch
import mlflow
from tqdm import tqdm
from pretrainData import PretrainTableDataset
from starmie.sdd.pretrain import train
import pandas as pd
import os
from Encodings import table_features
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="starmie")  # Valerie starmie
    parser.add_argument("--dataset", type=str, default="T2DV2")
    parser.add_argument("--logdir", type=str, default="model/")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--size", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--lm", type=str, default='roberta')
    parser.add_argument("--projector", type=int, default=768)
    parser.add_argument("--augment_op", type=str, default='shuffle_col,sample_row')
    parser.add_argument("--save_model", dest="save_model", action="store_true",default=True)
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    # single-column mode without table context
    parser.add_argument("--single_column", dest="single_column", action="store_true")
    parser.add_argument("--check_subject_Column",type=str, default='subjectheader')#subjectheader
    # row / column-ordered for preprocessing
    parser.add_argument("--table_order", type=str, default='sentence_row')  # column
    # for sampling
    parser.add_argument("--sample_meth", type=str, default='head')  # head tfidfrow
    # mlflow tag
    parser.add_argument("--mlflow_tag", type=str, default=None)

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
    path = 'datasets/%s/Test' % hp.dataset
    # trainset = PretrainTableDataset(path,
    #                      augment_op=hp.augment_op,
    #                      lm=hp.lm,
    #                      max_len=hp.max_len,
    #                      size=hp.size,
    #                      single_column=hp.single_column,
    #                      sample_meth=hp.sample_meth)
    trainset = PretrainTableDataset.from_hp(path, hp)
    trainset[1]
    """
    print(os.getcwd() + "/" + hp.logdir   + hp.method + "model_" + str(hp.augment_op) + "_" + str(
        hp.sample_meth) + "_" + str(hp.table_order) + '_' + str(hp.run_id) + "singleCol.pt")
    """

    #print(hp.save_model,hp.check_subject_Column)
    #train(trainset, hp)
    #table_features(hp)
    """
    total =None
    tables=[]
    for table in trainset.tables:
        table_csv = pd.read_csv(os.path.join(trainset.path, table),encoding="latin-1")
        tables.append(table_csv)
    tables = random.sample(tables, int(len(trainset.tables)*0.2))
    total = total if total is not None else len(tables)
    batch = []
    results = []
    for tid, table in tqdm(enumerate(tables), total=total):
        x, _ = trainset._tokenize(table)
        print(x,_)
        batch.append((x, x, []))
        
        
    print(batch)
  padder = trainset.pad(batch)
     """
