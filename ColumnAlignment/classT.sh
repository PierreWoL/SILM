#!/bin/bash --login
#$ -cwd
#$ -l nvidia_a100=1   # A 1-GPU request (v100 is just a shorter name for nvidia_v100)
                     # Can instead use 'a100' for the A100 GPUs (if permitted!)         --subject_column \ 'OpenData' 'TPC-DI'  'Magellan'
module load libs/cuda
conda activate py39
source /mnt/iusers01/fatpou01/compsci01/c29770zw/test/CurrentDataset/datavenv/bin/activate
parent_dir=$(dirname "$(pwd)")
datasets=('Wikidata' 'Magellan') 
for dataset in "${datasets[@]}"
  do
      subfolders=$(find "$parent_dir/ValentineDatasets/$dataset/" -maxdepth 1 -type d -name "*" ! -path "$parent_dir/ValentineDatasets/$dataset/" ! -path "$parent_dir/ValentineDatasets/$dataset/Train")
      for child_folder in $subfolders; do
        folder_name=$(basename "$child_folder")
        eval_path="$parent_dir/ValentineDatasets/$dataset/$folder_name"
        gt_path="$parent_dir/ValentineDatasets/$dataset/$folder_name/$folder_name""_mapping.json"
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
 


