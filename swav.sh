#DATASET_PATH="/path/to/imagenet/train"
EXPERIMENT_PATH="./experiments/swav_100ep_pretrain"
mkdir -p $EXPERIMENT_PATH

python -m torch.distributed.launch --nproc_per_node=1  E:/Project/CurrentDataset/SwAV.py \
--nmb_crops 2 6 \
--crops_for_assign 0 1 \
--temperature 0.1 \
--epsilon 0.01 \
--nmb_prototypes 3 \
--queue_length 0 \
--epochs 2 \
--batch_size 2 \
--wd 0.000001 \
--use_fp16 true \
--dump_path $EXPERIMENT_PATH
# --data_path $DATASET_PATH \
