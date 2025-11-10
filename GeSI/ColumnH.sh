#!/bin/bash --login
#SBATCH -p gpuA            
#SBATCH -G 2              
#SBATCH -t 1-0             
### Optional flags
#SBATCH -n 2    

source activate er_env  
module load apps/binapps/anaconda3/2024.10
module purge
module load libs/cuda


cd /mnt/iusers01/fatpou01/compsci01/c29770zw/ER2025-master/Summoner

python ColumnTypeHierachy.py