#!/bin/bash --login
#$ -cwd
#$ -l nvidia_v100=2   # A 1-GPU request (v100 is just a shorter name for nvidia_v100)
                     # Can instead use 'a100' for the A100 GPUs (if permitted!)    'OpenData' 'TPC-DI'
#$ -pe smp.pe 4  # 8 CPU cores available to the host code  'Joinable' 'Semantically-Joinable' 'Unionable' 'View-Unionable'  lm_path="$parent_dir/ColumnAlignment/model/$dataset""_FT_columnMatch/" $lm_path
module load libs/cuda
conda activate py39
source /mnt/iusers01/fatpou01/compsci01/c29770zw/test/CurrentDataset/datavenv/bin/activate
parent_dir=$(dirname "$(pwd)")
datasets=('ChEMBL' 'OpenData' 'TPC-DI')
for dataset in "${datasets[@]}"
  do
    for join in 'Unionable' 'Semantically-Joinable'; do
        train_path="$parent_dir/ValentineDatasets/$dataset/$join/Train"
        echo "train_path: $train_path"
    # obtain child folder list
        python main.py \
        --method M3 \
        --dataset $dataset \
        --Type $join \
        --batch 16 \
        --eval_batch_size 32 \
        --output_dir $train_path \
        --lm  roberta\
        --max_len 256 \
        --fine_tune \
        --save_model \
        --early_stop False \
        --early_stop_patience 10 \
        --positive_op random 
      done
  done
 


