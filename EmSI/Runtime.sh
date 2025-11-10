#!/bin/bash --login 
#$ -cwd
#$ -l v100=1
module load libs/cuda
conda activate py39
source /mnt/iusers01/fatpou01/compsci01/c29770zw/test/CurrentDataset/datavenv/bin/activate
for i in $(seq 1000 1000 10000)
do
python main.py --step 0 --dataset GoogleSearch --subjectCol --tableNumber $i --estimateNumber 40 --P23Embed cl_SCT8_lm_bert_head_column_0_none_column.pkl --P4Embed cl_SCT8_lm_bert_head_column_0_none_column.pkl
done