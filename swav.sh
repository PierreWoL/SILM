#!/bin/bash --login
#$ -cwd
#$ -l nvidia_v100=2
#$ -pe smp.pe 4

module load libs/cuda
conda activate py39
source /mnt/iusers01/fatpou01/compsci01/c29770zw/test/CurrentDataset/datavenv/bin/activate
module avail libs/nccl

nodelist=$(awk '{print $1}' $PE_HOSTFILE | tr '\n' ',' | sed 's/,$//')
echo "Node List: $nodelist"
master_node=$(head -n 1 $PE_HOSTFILE | awk '{print $1}')
echo "Master node: $master_node"
dist_url="tcp://$master_node:40000"
echo "Distributed URL: $dist_url"
#python -m torch.distributed.run --nproc_per_node=1 --rdzv_id=123 --rdzv_backend=c10d
selected_datasets=(12)
for num in "${selected_datasets[@]}"
  do
    EXPERIMENT_PATH="./model/swav/$num/"
  mkdir -p $EXPERIMENT_PATH
  python SwAV.py \
  --nmb_crops 2 2 \
  --datasetSize $num\
  --crops_for_assign 0 1 \
  --temperature 0.1 \
  --epsilon 0.01 \
  --nmb_prototypes 4 \
  --queue_length 20 \
  --epochs 1 \
  --batch_size 3 \
  --sinkhorn_iterations 3 \
  --wd 0.000001 \
  --use_fp16 true \
  --dump_path $EXPERIMENT_PATH
done




