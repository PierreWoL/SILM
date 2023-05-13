#!/bin/bash --login
#$ -cwd
#$ -l a100=2      # A 1-GPU request (v100 is just a shorter name for nvidia_v100)
                     # Can instead use 'a100' for the A100 GPUs (if permitted!)
#$ -pe smp.pe 4     # 8 CPU cores available to the host code
                     # Can use up to 12 CPUs with an A100 GPU. open_data
module load libs/cuda
conda activate py39
source /mnt/iusers01/fatpou01/compsci01/c29770zw/test/CurrentDataset/datavenv/bin/activate
datasets=("open_data")
check_subject_Columns=("subjectheader" "header" "none") 
table_orders=("sentence_row" "pure_row")  
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
        --batch_size 64 \
        --lr 5e-5 \
        --lm roberta \
        --n_epochs 64 \
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