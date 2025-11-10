#!/bin/bash --login
#$ -cwd
#$ -l nvidia_a100=1


module load libs/cuda/12.0.1
module load compilers/gcc/12.2.0

source activate test_env
nvcc --version
nvidia-smi

python main_vllm.py
#python try.py


