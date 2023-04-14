datasets=("open_data" "SOTAB" "T2DV2" "Test_corpus")
check_subject_Columns=("subjectheader" "subject" "none")
echo dataset check_subject_Column
for dataset in "${datasets[@]}"
do
  for check_subject_Column in "${check_subject_Columns[@]}"
  do
    echo $dataset $check_subject_Column
    python pretrain_all.py \
      --method starmie \
      --dataset $dataset \
      --batch_size 32 \
      --lr 5e-5 \
      --lm roberta \
      --n_epochs 20 \
      --max_len 256 \
      --size 10000 \
      --projector 768 \
      --save_model \
      --augment_op shuffle_col,shuffle_row \
      --fp16 \
      --check_subject_Column $check_subject_Column \
      --sample_meth head \
      --table_order column \
      --run_id 0
  done
done