#!/bin/bash --login
#$ -cwd
#$ -l nvidia_v100=1   # A 1-GPU request (v100 is just a shorter name for nvidia_v100)
                     # Can instead use 'a100' for the A100 GPUs (if permitted!)    'OpenData' 'TPC-DI'
#$ -pe smp.pe 4   # 8 CPU cores available to the host code 'Joinable' 'Unionable' 'View-Unionable' 'Semantically-Joinable'
module load libs/cuda
module load tools/bintools/git-lfs
conda activate py39
source /mnt/iusers01/fatpou01/compsci01/c29770zw/test/CurrentDataset/datavenv/bin/activate
HUGGINGFACE_TOKEN='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'
#huggingface-cli login --token $HUGGINGFACE_TOKEN
python tryCM.py

   


