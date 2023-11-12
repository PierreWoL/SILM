#!/bin/bash --login
#$ -cwd
#$ -l nvidia_a100=1   # A 1-GPU request (v100 is just a shorter name for nvidia_v100)
                     # Can instead use 'a100' for the A100 GPUs (if permitted!)    'OpenData' 'TPC-DI'
#$ -pe smp.pe 2   # 8 CPU cores available to the host code 'Joinable' 'Unionable' 'View-Unionable' 'Semantically-Joinable' lm_path="$parent_dir/ColumnAlignment/model/$dataset""_FT_columnMatch/" $lm_path
module load libs/cuda
conda activate py39
source /mnt/iusers01/fatpou01/compsci01/c29770zw/test/CurrentDataset/datavenv/bin/activate
parent_dir=$(dirname "$(pwd)")
datasets=('ChEMBL' 'OpenData' 'TPC-DI')
for dataset in "${datasets[@]}"
  do
    for join in 'View-Unionable'; do
    # obtain child folder list
      subfolders=$(find "$parent_dir/ValentineDatasets/$dataset/$join" -maxdepth 1 -type d -name "*" ! -path "$parent_dir/ValentineDatasets/$dataset/$join" ! -path "$parent_dir/ValentineDatasets/$dataset/$join/Train")
      for child_folder in $subfolders; do
        folder_name=$(basename "$child_folder")
        eval_path="$parent_dir/ValentineDatasets/$dataset/$join/$folder_name"
        gt_path="$parent_dir/ValentineDatasets/$dataset/$join/$folder_name/$folder_name""_mapping.json"
        lm_path="$parent_dir/ValentineDatasets/$dataset/$join/Train/$dataset""_FT_columnMatch/"
        echo "eval_path: $eval_path"
        echo "gt_path: $gt_path"
        echo "lm_path: $lm_path"
        python Main2.py \
        --method M3 \
        --dataset $dataset \
        --Type $join \
        --ground_truth_path  $gt_path \
        --eval_path   $eval_path \
        --batch 32 \
        --eval_batch_size 8 \
        --lm  $lm_path\
        --max_len 256 \
        --early_stop False \
        --early_stop_patience 10 \
        --positive_op random   
      done
  done
done
 


