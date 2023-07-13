#!/bin/bash --login
#$ -cwd
#$ -l v100=1       # A 1-GPU request (v100 is just a shorter name for nvidia_v100)
                     # Can instead use 'a100' for the A100 GPUs (if permitted!)

                     # Can use up to 12 CPUs with an A100 GPU.
module load libs/cuda
conda activate py39
source /mnt/iusers01/fatpou01/compsci01/c29770zw/test/CurrentDataset/datavenv/bin/activate
python tableEncoding.py
