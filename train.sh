#!/bin/bash --login
#$ -cwd 
#$ -l a100=1         # A 1-GPU request (v100 is just a shorter name for nvidia_v100)
                     # Can instead use 'a100' for the A100 GPUs (if permitted!)
                     # 8 CPU cores available to the host code
                     # Can use up to 12 CPUs with an A100 GPU.
module load libs/cuda
conda activate py39
source /mnt/iusers01/fatpou01/compsci01/c29770zw/test/CurrentDataset/datavenv/bin/activate
datasets=("WDC" "GDS" "GoogleSearch")
check_subject_Columns=("none" "header")
models=("sbert" "bert" "roberta")
augmentation=("sample_cells" "sample_cells_TFIDF") #"sample_cells"  sample_cells_TFIDF
aug_times=(7) #1
for dataset in "${datasets[@]}"
do
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
                    if (( time==7 )); then
                      batch=7
                      epoches=10
                    elif (( time==8 )); then
                      batch=1
                      epoches=10
                    fi
                  done
                  echo $dataset $check_subject_Column $model "$time $augment_methods batch $batch poches $epoches" 
                  python pretrain_all.py \
                  --method starmie \
                  --dataset $dataset \
                  --batch_size $batch \
                  --lr 5e-5 \
                  --lm $model \
                  --fp16 \
                  --n_epochs $epoches \
                  --max_len 256 \
                  --size 10000 \
                  --projector 768 \
                  --save_model \
                  --column \
                  --augment_op $augment_methods \
                  --check_subject_Column $check_subject_Column \
                  --sample_meth head \
                  --table_order column \
                  --run_id 0
          done
        done
      done
  done
done