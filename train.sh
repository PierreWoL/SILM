#!/bin/bash --login
#$ -cwd 
#$ -l a100=2         # A 1-GPU request (v100 is just a shorter name for nvidia_v100)
                     # Can instead use 'a100' for the A100 GPUs (if permitted!)
                     # 8 CPU cores available to the host code
                     # Can use up to 12 CPUs with an A100 GPU. --datasetSize $num \
#$ -m bea
#$ -M zhenyu.wu@manchester.ac.uk

module load libs/cuda
conda activate py39
source /mnt/iusers01/fatpou01/compsci01/c29770zw/test/CurrentDataset/datavenv/bin/activate
datasets=("GDS" "WDC")
check_subject_Columns=("none")
models=("sbert")
augmentation=("sample_cells_TFIDF") #"sample_cells"  sample_cells_TFIDF  "header" --column \
aug_times=(4 6 8) #1
selected_datasets=(-1)
for dataset in "${datasets[@]}"
do
#for num in "${selected_datasets[@]}"
#do
  for check_subject_Column in "${check_subject_Columns[@]}"
  do
    for model in "${models[@]}"
      do
      for aug in "${augmentation[@]}"
        do
          for time in "${aug_times[@]}"
              do
              augment_methods=""
              for ((i=1; i<=time; i++))
                 do
                    if [[ $i -eq $time ]]; then
                        augment_methods+="$aug"
                    else
                        augment_methods+="$aug,"
                    fi
                    if (( time>=6 )); then
                      batch=8
                      epoches=10
                    elif (( time==4 )); then
                      batch=12
                      epoches=20
                    else
                        batch=64
                        epoches=35
                    fi
                  done
                  echo $dataset $check_subject_Column $model "$time $num $augment_methods batch $batch epoch $epoches" 
                  python pretrain_all.py \
                  --dataset $dataset \
                  --batch_size $batch \
                  --lm $model \
                  --fp16 \
                  --n_epochs $epoches \
                  --save_model \
                  --subject_column \
                  --augment_op $augment_methods \
                  --check_subject_Column $check_subject_Column \
                  --sample_meth head \
                  --table_order column \
                  --run_id 0
          done
        done
      done
  done
 # done
done