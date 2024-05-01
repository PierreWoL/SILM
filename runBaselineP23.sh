#!/bin/bash --login
#$ -cwd
#$ -l v100=1
module load libs/cuda
conda activate py39
source /mnt/iusers01/fatpou01/compsci01/c29770zw/test/CurrentDataset/datavenv/bin/activate
models=("glove")
datasets=("GDS" "WDC")
for data in "${datasets[@]}"
do
for model in "${models[@]}"
      do
      echo $model
      python main.py \
      --baseline \
      --dataset $data \
      --intervalSlice 10 \
      --step 2 \
      --embed $model
      done
done