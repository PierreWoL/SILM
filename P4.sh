#!/bin/bash --login
#$ -cwd
module load libs/cuda
conda activate py39
source /mnt/iusers01/fatpou01/compsci01/c29770zw/test/CurrentDataset/datavenv/bin/activate

models=("DP")
datasets=("WDC")
for data in "${datasets[@]}"
do
for i in $(seq 0.8 0.1 0.8)
do
  for p in $(seq 0.1 0.1 0.9)
  do
    for lm in "${models[@]}"
      do
        echo $i $lm $p
     python main.py --step 4 --dataset $data --similarity $i --portion $p --embed $lm
      done
    done
done
done