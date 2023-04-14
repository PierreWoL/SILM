import argparse
import os
from Encodings import starmie_clustering
import numpy as np
import random
import torch
import mlflow
from experiments import run_exp

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
    parser.add_argument("--save_model", dest="save_model", action="store_true", default=True)
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    # single-column mode without table context
    parser.add_argument("--single_column", dest="single_column", action="store_true")
    parser.add_argument("--is_sub", type=bool, default=True)
    parser.add_argument("--check_subject_Column", type=str, default='subjectheader')  # subjectheader
    # row / column-ordered for preprocessing
    parser.add_argument("--table_order", type=str, default='column')  # column
    # for sampling
    parser.add_argument("--sample_meth", type=str, default='head')  # head tfidfrow
    # mlflow tag


    hp = parser.parse_args()
    method = 0
    starmie_clustering(hp)
    # trainset = PretrainTableDataset(path,
    #                      augment_op=hp.augment_op,
    #                      lm=hp.lm,
    #                      max_len=hp.max_len,
    #                      size=hp.size,
    #                      single_column=hp.single_column,
    #                      sample_meth=hp.sample_meth)
    # trainset = PretrainTableDataset.from_hp(path, hp)

