#!/bin/bash --login
#$ -cwd
#$ -l nvidia_a100=1   # A 1-GPU request (v100 is just a shorter name for nvidia_v100)
#'OpenData' 'TPC-DI' 'ChEMBL' 'Unionable' 'Joinable'   'Semantically-Joinable'  'View-Unionable'
module load libs/cuda
conda activate py39
source /mnt/iusers01/fatpou01/compsci01/c29770zw/test/CurrentDataset/datavenv/bin/activate
parent_dir=$(dirname "$(pwd)")
datasets=('TPC-DI')
for dataset in "${datasets[@]}"
  do
    for join in 'Joinable' 'Semantically-Joinable' 'View-Unionable' 'Unionable'; do
    # obtain child folder list
      subfolders=$(find "$parent_dir/ValentineDatasets/$dataset/$join" -maxdepth 1 -type d -name "*" ! -path "$parent_dir/ValentineDatasets/$dataset/$join")
      for child_folder in $subfolders; do
        folder_name=$(basename "$child_folder")
        eval_path="$parent_dir/ValentineDatasets/$dataset/$join/$folder_name"
        gt_path="$parent_dir/ValentineDatasets/$dataset/$join/$folder_name/$folder_name""_mapping.json"
        echo "eval_path: $eval_path"
        echo "gt_path: $gt_path"
        python main.py \
        --method M2 \
        --dataset $dataset \
        --ground_truth_path  $gt_path \
        --eval_path   $eval_path \
        --batch 16 \
        --eval_batch_size 16 \
        --lm roberta \
        --num_epochs 10 \
        --max_len 256 \
        --early_stop False \
        --early_stop_patience 10 \
        --save_model \
        --positive_op random
      done
  done
done
 


