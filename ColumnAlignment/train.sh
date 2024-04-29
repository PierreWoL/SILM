#!/bin/bash --login
#$ -cwd
#$ -l nvidia_a100=2   # A 1-GPU request (v100 is just a shorter name for nvidia_v100)
                     # Can instead use 'a100' for the A100 GPUs (if permitted!)         --subject_column \
#$ -pe smp.pe 2   # 8 CPU cores available to the host code "subjectheader" "header" "none"  "WDC"
                     # lm model/OpenData_FT_columnMatch 'ChEMBL' 'TPC-DI' 'OpenData'  'Wikidata' 'Magellan'
module load libs/cuda
conda activate py39
source /mnt/iusers01/fatpou01/compsci01/c29770zw/test/CurrentDataset/datavenv/bin/activate
parent_dir=$(dirname "$(pwd)")
datasets=('Wikidata' 'Magellan')
for dataset in "${datasets[@]}"
  do
        echo "eval_path: $eval_path"
        echo "gt_path: $gt_path"
        echo "pair_name: $folder_name"
          python main.py \
        --method M3 \
        --dataset $dataset \
        --batch 16 \
        --eval_batch_size 16 \
        --lm roberta \
        --num_epochs 20 \
        --max_len 256 \
        --early_stop False \
        --early_stop_patience 10 \
        --save_model \
        --fine_tune \
        --positive_op random
done