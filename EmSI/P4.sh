#!/bin/bash --login
#$ -cwd     # For 32GB per core, any of the CPU types below (system chooses)
#$ -l mem512

module load libs/cuda
conda activate py39
source /mnt/iusers01/fatpou01/compsci01/c29770zw/test/CurrentDataset/datavenv/bin/activate

models=("sbert")
datasets=("WDC")
for data in "${datasets[@]}"
do
for i in $(seq 0.8 0.1 0.9)
do
  for p in $(seq 0.6 0.1 0.9)
  do
for psa in $(seq 0.05 0.05 0.25)
do   
 for lm in "${models[@]}"
      do
        echo $i $lm $p $psa
     python main.py --step 4 --dataset $data --similarity $i --portion $p --portionSA $psa --embed $lm --baseline
      done
done
done   
 done
done
