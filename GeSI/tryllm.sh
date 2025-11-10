#!/bin/bash --login
#$ -cwd
#$ -l nvidia_a100=1


module load libs/cuda/12.0.1
module load compilers/gcc/12.2.0
source activate test_env
nvcc --version
nvidia-smi

ollama serve
sleep 10  

#OLLAMA_USE_CUDA=1 ollama run deepseek-r1:14b "hi, tell us a joke" > ollama_test_output.log 2>&1



python mainFET.py
echo "Test completed. Check ollama_test_output.log for results."
