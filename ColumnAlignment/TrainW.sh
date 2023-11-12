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
        train_path="$parent_dir/ValentineDatasets/$dataset/Train"
        echo "train_path: $train_path"
        python main.py \
        --method M3 \
        --dataset $dataset \
        --batch 32 \
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
 