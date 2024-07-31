#!/bin/bash --login
#$ -cwd
#$ -l nvidia_a100=2
#$ -pe smp.pe 3
#$ -ac nvmps

module load mpi/intel-18.0/openmpi/4.0.1-cuda
module load libs/cuda/11.0.3
conda activate py39
source /mnt/iusers01/fatpou01/compsci01/c29770zw/test/CurrentDataset/datavenv/bin/activate
module load libs/nccl/2.8.3
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
echo "Job is using $NGPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $NSLOTS CPU core(s)"


nodelist=$(awk '{print $1}' $PE_HOSTFILE | tr '\n' ',' | sed 's/,$//')
echo "Node List: $nodelist"
master_node=$(head -n 1 $PE_HOSTFILE | awk '{print $1}')
echo "Master node: $master_node"
dist_url="tcp://$master_node:40000"
echo "Distributed URL: $dist_url"

num_nodes=$(wc -l < $PE_HOSTFILE)
world_size=$((num_nodes * NGPUS))
echo "world_size: $world_size "
export TORCH_DISTRIBUTED_DEBUG=DETAIL


selected_datasets=(-1 200 400)
for size in "${selected_datasets[@]}"
do
EXPERIMENT_PATH="./model/swav/WDC/$size/"
mkdir -p $EXPERIMENT_PATH
mpirun -np $world_size python -u SwAV.py \
--world_size $world_size \
--datasetSize $size \
--nmb_crops 2 4 \
--lm sbert \
--crops_for_assign 0 1 \
--temperature 0.1 \
--epsilon 0.03 \
--subject_column \
--nmb_prototypes 300 \
--queue_length 30 \
--epochs 45 \
--batch_size 20 \
--sinkhorn_iterations 3 \
--wd 0.000001 \
--use_fp16 true \
--column \
--dump_path $EXPERIMENT_PATH
done
#python -m torch.distributed.run --nproc_per_node=1 --rdzv_id=123 --rdzv_backend=c10d 

    



 
