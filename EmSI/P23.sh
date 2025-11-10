#!/bin/bash --login
#$ -cwd
#$ -l mem512     # For 32GB per core, any of the CPU types below (system chooses)

module load libs/cuda
conda activate py39
source /mnt/iusers01/fatpou01/compsci01/c29770zw/test/CurrentDataset/datavenv/bin/activate
deltas=(0.15)
datasets=("WDC" "GDS")
models=("sbert")
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
      --step 2 \
      --intervalSlice 10 \
      --delta $delta
    done
done
done
      
 