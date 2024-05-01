#!/bin/bash --login
#$ -cwd
#$ -l a100=1
module load libs/cuda
conda activate py39
source /mnt/iusers01/fatpou01/compsci01/c29770zw/test/CurrentDataset/datavenv/bin/activate

deltas=(0.1 0.15 0.2)
datasets=("GDS" "WDC")
models=("roberta" "sbert" "bert")
for data in "${datasets[@]}"
do
for model in "${models[@]}"
do
      for delta in "${deltas[@]}"
      do
      echo $model   $delta
      python main.py \
      --embed $model \
      --dataset $data \
      --step 3 \
      --slice_start 0 \
      --slice_stop 10 \
      --delta $delta
    done
done 
done
 