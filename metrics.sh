datasets=("open_data" "SOTAB" "T2DV2" "Test_corpus")
is_sub=("True" "False")
echo dataset check_subject_Column
for dataset in "${datasets[@]}"
do
  for check_subject_Column in "${is_sub[@]}"
  do
    echo $dataset $check_subject_Column
    python pretrain_all.py \
      --method starmie \
      --dataset $dataset \
      --check_subject_Column $check_subject_Column
  done
done