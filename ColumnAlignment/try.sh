#!/bin/bash --login
#$ -cwd
#$ -l nvidia_v100=2   # A 1-GPU request (v100 is just a shorter name for nvidia_v100)
                     # Can instead use 'a100' for the A100 GPUs (if permitted!)         --subject_column \
#$ -pe smp.pe 8   # 8 CPU cores available to the host code "subjectheader" "header" "none"  "WDC"
                     # lm model/OpenData_FT_columnMatch
parent_dir=$(dirname "$(pwd)")
datasets=('ChEMBL' 'OpenData' 'TPC-DI')
for dataset in "${datasets[@]}"
  do
    for join in 'Joinable' 'Semantically-Joinable' 'Unionable' 'View-Unionable'; do
    # obtain child folder list
      subfolders=$(find "$parent_dir/ValentineDatasets/$dataset/$join" -maxdepth 1 -type d -name "*" ! -path "$parent_dir/ValentineDatasets/$dataset/$join")
      for child_folder in $subfolders; do
        folder_name=$(basename "$child_folder")
        # ¹¹½¨Â·¾¶
        eval_path="$parent_dir/ValentineDatasets/$dataset/$join/$folder_name"
        gt_path="$parent_dir/ValentineDatasets/$dataset/$join/$folder_name/$folder_name""_mapping.json"
        echo "eval_path: $eval_path"
        echo "gt_path: $gt_path"
        echo "pair_name: $folder_name"
      done
  done
done