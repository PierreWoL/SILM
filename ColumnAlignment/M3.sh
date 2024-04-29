#!/bin/bash --login
#$ -cwd
#$ -l nvidia_v100=1   # A 1-GPU request (v100 is just a shorter name for nvidia_v100)
                     # Can instead use 'a100' for the A100 GPUs (if permitted!)         --subject_column \ 'OpenData' 'TPC-DI'
   # 8 CPU cores available to the host code "subjectheader" "header" "none"  "WDC"
module load libs/cuda
conda activate py39
source /mnt/iusers01/fatpou01/compsci01/c29770zw/test/CurrentDataset/datavenv/bin/activate
parent_dir=$(dirname "$(pwd)")
datasets=('Magellan' 'Wikidata') 
for dataset in "${datasets[@]}"
  do
      subfolders=$(find "$parent_dir/ValentineDatasets/$dataset/" -maxdepth 1 -type d -name "*" ! -path "$parent_dir/ValentineDatasets/$dataset/" ! -path "$parent_dir/ValentineDatasets/$dataset/Train")
      for child_folder in $subfolders; do
        folder_name=$(basename "$child_folder")
        eval_path="$parent_dir/ValentineDatasets/$dataset/$folder_name"
        gt_path="$parent_dir/ValentineDatasets/$dataset/$folder_name/$folder_name""_mapping.json"
        lm_path="$parent_dir/ColumnAlignment/model/$dataset""_FT_columnMatch/"
        echo "eval_path: $eval_path"
        echo "gt_path: $gt_path"
        python main.py \
        --method M3 \
        --dataset $dataset \
        --ground_truth_path  $gt_path \
        --eval_path   $eval_path \
        --batch 32 \
        --eval_batch_size 8 \
        --lm roberta \
        --max_len 256 \
        --early_stop False \
        --early_stop_patience 10 \
        --positive_op random \
        --fine_tune \
        --save_model
  done
done
 


