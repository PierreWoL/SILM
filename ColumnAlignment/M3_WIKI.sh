#!/bin/bash --login
#$ -cwd
#$ -l nvidia_v100=1   # A 1-GPU request (v100 is just a shorter name for nvidia_v100)
                     # Can instead use 'a100' for the A100 GPUs (if permitted!)         --subject_column \ 'OpenData' 'TPC-DI'
   # 8 CPU cores available to the host code "subjectheader" "header" "none"  "WDC"
module load libs/cuda
conda activate py39
source /mnt/iusers01/fatpou01/compsci01/c29770zw/test/CurrentDataset/datavenv/bin/activate
parent_dir=$(dirname "$(pwd)")
folders=('musicians_semjoinable' 'musicians_unionable' 'musicians_joinable') 
for folder in "${folders[@]}"
  do
  eval_path="$parent_dir/ValentineDatasets/Wikidata/$folder"
  gt_path="$parent_dir/ValentineDatasets/Wikidata/$folder/$folder""_mapping.json"
  lm_path="$parent_dir/ColumnAlignment/model/Wikidata_FT_columnMatch/"
  python main.py \
        --method M3 \
        --dataset Wikidata\
        --ground_truth_path $gt_path \
        --eval_path $eval_path \
        --batch 16 \
        --eval_batch_size 8 \
        --lm $lm_path \
        --max_len 256 \
        --early_stop False \
        --early_stop_patience 10 \
        --positive_op random  
  done