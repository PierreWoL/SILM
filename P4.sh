#!/bin/bash --login
#$ -cwd
#$ -l mem512     # For 32GB per core, any of the CPU types below (system chooses)

module load libs/cuda
conda activate py39
source /mnt/iusers01/fatpou01/compsci01/c29770zw/test/CurrentDataset/datavenv/bin/activate

models=("sbert")
datasets=("GDS")
for data in "${datasets[@]}"
do
for i in $(seq 0.5 0.1 0.9)
do
  for p in $(seq 0.1 0.1 0.9)
  do
    for lm in "${models[@]}"
      do
        echo $i $lm $p
     python main.py --step 4 --dataset $data --similarity $i --portion $p --embed $lm --baseline
      done
    done
done
done