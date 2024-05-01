#!/bin/bash --login
#$ -cwd

module load libs/cuda
conda activate py39
source /mnt/iusers01/fatpou01/compsci01/c29770zw/test/CurrentDataset/datavenv/bin/activate
datasets=("GoogleSearch" "WDC" "GDS")
check_subject_Columns=("none" "header")
models=("bert" "sbert" "roberta")
for dataset in "${datasets[@]}"
do
  for check_subject_Column in "${check_subject_Columns[@]}"
  do
    for model in "${models[@]}"
      do
        echo $dataset $check_subject_Column $model 
        python pretrain_all.py \
        --dataset $dataset \
        --lm $model \
        --pretrain \
        --check_subject_Column $check_subject_Column
      done
  done
done