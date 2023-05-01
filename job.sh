#!/bin/bash --login
#$ -cwd
#$ -l a100           # A 1-GPU request (v100 is just a shorter name for nvidia_v100)
                     # Can instead use 'a100' for the A100 GPUs (if permitted!)
#$ -pe smp.pe 4     # 8 CPU cores available to the host code
                     # Can use up to 12 CPUs with an A100 GPU.

# Latest version of CUDA
module load libs/cuda
conda activate py39
source /mnt/iusers01/fatpou01/compsci01/c29770zw/test/CurrentDataset/datavenv/bin/activate
datasets=("open_data" "WDC")
check_subject_Columns=("subjectheader" "header" "none")
table_orders=("column") #"pure_row" "sentence_row"
echo dataset check_subject_Column
for dataset in "${datasets[@]}"
do
  for check_subject_Column in "${check_subject_Columns[@]}"
  do
    for table_order in "${table_orders[@]}"
      do
        echo $dataset $check_subject_Column $table_order
        python pretrain_all.py \
        --method starmie \
        --dataset $dataset \
        --batch_size 32 \
        --lr 5e-5 \
        --lm sbert \
        --n_epochs 20 \
        --max_len 256 \
        --size 10000 \
        --projector 768 \
        --save_model \
        --augment_op drop_num_col \
        --fp16 \
        --check_subject_Column $check_subject_Column \
        --sample_meth head \
        --table_order $table_order \
        --run_id 0
      done
  done
done